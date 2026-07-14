"""Full-corpus SBCSAE normalization with profiles and quality diagnostics."""

from __future__ import annotations

import csv
import hashlib
import re
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .data import clean_transcript_text, parse_trn
from .utils import write_json, write_jsonl

UNKNOWN_MARKERS = {"", "?", "UNKNOWN", "N/A", "NA", "NONE", "NIL"}
NON_PERSON_NAMES = {
    "ALL",
    "AUD",
    "BABY",
    "BOTH",
    "CAT",
    "CONGR",
    "DOG",
    "ENV",
    "HORSE",
    "MANY",
    "RADIO",
    "READ",
    "X",
}
NON_PERSON_RE = re.compile(r"^(?:AUD|X)(?:_?\d+)?$")
JOINT_SPEAKER_RE = re.compile(r"[/+&]")
SPEAKER_ALIASES = {
    "JANICD": "JANICE",
    "MONTOYA": "MONTOYO",
}

# High-confidence disambiguation for names that identify different people in
# different conversations. Values are metadata participant IDs or source rows.
PROFILE_OVERRIDES = {
    ("SBC001", "LENORE"): "metadata1.csv:2",
    ("SBC001", "DORIS"): "metadata1.csv:3",
    ("SBC002", "JAMIE"): "metadata1.csv:6",
    ("SBC004", "CAROLYN"): "metadata1.csv:11",
    ("SBC004", "KATHY"): "metadata1.csv:12",
    ("SBC004", "SHANE"): "metadata1.csv:14",
    ("SBC006", "LENORE"): "0001",
    ("SBC007", "ALICE"): "metadata1.csv:20",
    ("SBC007", "MARY"): "metadata1.csv:21",
    ("SBC009", "KATHY"): "metadata1.csv:26",
    ("SBC010", "BRAD"): "metadata1.csv:28",
    ("SBC010", "PHIL"): "metadata1.csv:29",
    ("SBC011", "DORIS"): "metadata1.csv:30",
    ("SBC012", "CAROLYN"): "metadata1.csv:37",
    ("SBC012", "LAURA"): "metadata1.csv:38",
    ("SBC012", "FRANK"): "metadata1.csv:39",
    ("SBC013", "KENDRA"): "metadata1.csv:42",
    ("SBC013", "KEN"): "metadata1.csv:43",
    ("SBC013", "KEVIN"): "metadata1.csv:46",
    ("SBC014", "JIM"): "metadata1.csv:47",
    ("SBC014", "FRED"): "metadata1.csv:48",
    ("SBC015", "KEN"): "0053",
    ("SBC015", "LENORE"): "0001",
    ("SBC016", "BRAD"): "0054",
    ("SBC017", "JIM"): "0056",
    ("SBC019", "FRANK"): "0064",
    ("SBC020", "JEFF"): "0068",
    ("SBC020", "DAN"): "0070",
    ("SBC022", "BILL"): "0076",
    ("SBC023", "LINDA"): "0077",
    ("SBC023", "LORI"): "0083",
    ("SBC023", "PATTY"): "0086",
    ("SBC024", "DAN"): "0071",
    ("SBC024", "JENNIFER"): "0090",
    ("SBC027", "PHIL"): "0094",
    ("SBC028", "JEFF"): "0069",
    ("SBC031", "JAMIE"): "0102",
    ("SBC033", "BILL"): "0110",
    ("SBC033", "LAURA"): "0111",
    ("SBC033", "MARY"): "0114",
    ("SBC033", "RICHARD"): "0115",
    ("SBC033", "DON"): "0112",
    ("SBC034", "KAREN"): "0116",
    ("SBC035", "PATTY"): "0118",
    ("SBC036", "KEVIN"): "0126",
    ("SBC037", "SHANE"): "0130",
    ("SBC039", "ALICE"): "0139",
    ("SBC039", "DON"): "0136",
    ("SBC039", "LORI"): "0137",
    ("SBC042", "KENDRA"): "0143",
    ("SBC043", "ALICE"): "0149",
    ("SBC043", "ANNETTE"): "0150",
    ("SBC046", "DARREN"): "0156",
}

# The first experiment uses these 16 core dyadic conversations.  Their CHAT
# comments are short enough to review directly, so relationship/situation should
# not be inferred by a broad keyword rule (for example, "meeting" incorrectly
# turned an engineer/customer estimate into colleagues).  Values below were
# checked against the official @Comment text bundled with SBCSAE.
CORE_CONTEXT_OVERRIDES = {
    "SBC005": ("romantic_partners", "casual_social_conversation"),
    "SBC006": ("family", "casual_social_conversation"),
    "SBC007": ("family", "casual_social_conversation"),
    "SBC009": ("romantic_partners", "collaborative_task"),
    "SBC010": ("colleagues", "workplace_or_business"),
    "SBC017": ("friends_or_peers", "casual_social_conversation"),
    "SBC024": ("romantic_partners", "collaborative_task"),
    "SBC029": ("professional_client", "workplace_or_business"),
    "SBC034": ("romantic_partners", "family_or_home_conversation"),
    "SBC041": ("professional_client", "healthcare_consultation"),
    "SBC043": ("family", "family_or_home_conversation"),
    "SBC044": ("friends_or_peers", "casual_social_conversation"),
    "SBC045": ("friends_or_peers", "casual_social_conversation"),
    "SBC047": ("family", "family_or_home_conversation"),
    "SBC058": ("family", "family_or_home_conversation"),
    "SBC060": ("colleagues", "casual_social_conversation"),
}


@dataclass(frozen=True)
class MetadataRecord:
    record_key: str
    participant_id: str
    name: str
    gender: str
    age: str
    hometown: str
    home_state: str
    current_state: str
    education: str
    years_education: str
    occupation: str
    ethnicity: str
    conversation_hint: str
    source_file: str
    source_line: int


@dataclass(frozen=True)
class ChatParticipant:
    code: str
    display_name: str
    role: str


def normalize_value(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.replace("\x1e", "-")).strip(" \t\r\n\"")
    return "unknown" if cleaned.upper() in UNKNOWN_MARKERS else cleaned


def normalize_speaker_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip().upper()
    cleaned = cleaned.lstrip("#*>").rstrip("?").strip()
    return SPEAKER_ALIASES.get(cleaned, cleaned)


def metadata_aliases(value: str) -> set[str]:
    normalized = normalize_speaker_name(re.sub(r"\s*\([^)]*\)\s*", "", value))
    aliases = {normalized}
    for part in re.split(r"[/]", normalized):
        aliases.add(part.strip())
    tokens = normalized.split()
    if len(tokens) > 1:
        aliases.add(tokens[-1])
    if normalized == "ANETTE":
        aliases.add("ANNETTE")
    return {alias for alias in aliases if alias}


def is_person_name(name: str, role: str = "Speaker") -> bool:
    normalized = normalize_speaker_name(name)
    if role.lower() == "environment":
        return False
    if normalized in NON_PERSON_NAMES or NON_PERSON_RE.fullmatch(normalized):
        return False
    return not bool(JOINT_SPEAKER_RE.search(normalized))


def age_group(age: str) -> str:
    normalized = normalize_value(age)
    try:
        years = int(float(normalized))
    except ValueError:
        return "unknown"
    if years < 13:
        return "0-12"
    if years < 18:
        return "13-17"
    if years < 25:
        return "18-24"
    if years < 35:
        return "25-34"
    if years < 45:
        return "35-44"
    if years < 55:
        return "45-54"
    if years < 65:
        return "55-64"
    return "65+"


def _decode_csv_line(raw_line: str) -> list[str]:
    values = next(csv.reader([raw_line]))
    if len(values) == 1 and "," in values[0]:
        values = next(csv.reader([values[0]]))
    return [value.strip() for value in values]


def load_metadata(metadata_dir: str | Path) -> tuple[list[MetadataRecord], list[dict[str, Any]]]:
    records: list[MetadataRecord] = []
    issues: list[dict[str, Any]] = []
    for path in sorted(Path(metadata_dir).glob("metadata*.csv")):
        raw_text = path.read_text(encoding="utf-8-sig", errors="replace")
        control_repairs = raw_text.count("\x1e")
        if control_repairs:
            issues.append(
                {
                    "scope": "metadata",
                    "source_file": path.name,
                    "reason": "control_character_normalized",
                    "count": control_repairs,
                }
            )
        conversation_hint = ""
        for line_number, raw_line in enumerate(raw_text.replace("\x1e", "-").splitlines(), 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("*sbc"):
                conversation_hint = stripped[1:].upper()
                continue
            values = _decode_csv_line(stripped)
            if values and values[0].upper() == "NAME":
                continue
            if path.stem == "metadata1":
                values = [""] + values
            while values and not values[-1]:
                values.pop()
            if len(values) < 11:
                issues.append(
                    {
                        "scope": "metadata",
                        "source_file": path.name,
                        "line_number": line_number,
                        "reason": "too_few_fields",
                        "field_count": len(values),
                    }
                )
                values += [""] * (11 - len(values))
            if len(values) > 11:
                issues.append(
                    {
                        "scope": "metadata",
                        "source_file": path.name,
                        "line_number": line_number,
                        "reason": "extra_fields",
                        "field_count": len(values),
                    }
                )
                values = values[:9] + values[-2:]
            participant_id = values[0].strip()
            record_key = participant_id or f"{path.name}:{line_number}"
            records.append(
                MetadataRecord(
                    record_key=record_key,
                    participant_id=participant_id,
                    name=normalize_speaker_name(values[1]),
                    gender=normalize_value(values[2]).lower(),
                    age=normalize_value(values[3]),
                    hometown=normalize_value(values[4]),
                    home_state=normalize_value(values[5]),
                    current_state=normalize_value(values[6]),
                    education=normalize_value(values[7]),
                    years_education=normalize_value(values[8]),
                    occupation=normalize_value(values[9]),
                    ethnicity=normalize_value(values[10]),
                    conversation_hint=conversation_hint,
                    source_file=path.name,
                    source_line=line_number,
                )
            )
    return records, issues


def parse_chat_headers(path: str | Path) -> tuple[list[ChatParticipant], list[str], str]:
    participants: list[ChatParticipant] = []
    comments: list[str] = []
    media = ""
    current_comment: int | None = None
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if raw_line.startswith("@Participants:"):
            payload = raw_line.split(":", 1)[1]
            for item in payload.split(","):
                tokens = item.strip().split()
                if len(tokens) >= 3:
                    participants.append(
                        ChatParticipant(tokens[0], " ".join(tokens[1:-1]), tokens[-1])
                    )
            current_comment = None
        elif raw_line.startswith("@Comment:"):
            comments.append(raw_line.split(":", 1)[1].strip())
            current_comment = len(comments) - 1
        elif raw_line.startswith("@Media:"):
            media = raw_line.split(":", 1)[1].split(",", 1)[0].strip()
            current_comment = None
        elif raw_line.startswith("\t") and current_comment is not None:
            comments[current_comment] = f"{comments[current_comment]} {raw_line.strip()}".strip()
        elif raw_line.startswith("@"):
            current_comment = None
        elif raw_line.startswith("*"):
            break
    return participants, comments, media


def infer_context(context: str) -> tuple[str, str, str]:
    text = context.lower()
    if any(word in text for word in ("medical", "dietician", "orthopedist", "veterinarian")):
        situation = "healthcare_consultation"
    elif any(word in text for word in ("sermon", "church", "theology")):
        situation = "religious_or_spiritual_event"
    elif any(word in text for word in ("lecture", "classroom", "class ", "training", "tour ", "storytelling")):
        situation = "education_or_public_presentation"
    elif any(word in text for word in ("court", "city meeting", "public forum", "public lecture")):
        situation = "civic_or_legal_event"
    elif any(word in text for word in ("business", "meeting", "sales", "bank", "office", "workplace")):
        situation = "workplace_or_business"
    elif any(word in text for word in ("telephone", "phone conversation")):
        situation = "telephone_conversation"
    elif any(word in text for word in ("task-related", "task related", "game-playing", "cooking")):
        situation = "collaborative_task"
    elif any(word in text for word in ("family", "home", "christmas", "birthday")):
        situation = "family_or_home_conversation"
    else:
        situation = "casual_social_conversation"

    if any(word in text for word in ("patient", "consulting", "sales encounter", "attorney", "witness")):
        relationship = "professional_client"
    elif any(word in text for word in ("training", "classroom", "professor", "teacher", "instructor")):
        relationship = "teacher_student"
    elif any(word in text for word in ("audience", "lecture", "sermon", "tour", "storytelling")):
        relationship = "speaker_audience"
    elif any(
        word in text
        for word in (
            "family",
            "sister",
            "brother",
            "mother",
            "father",
            "cousin",
            "relatives",
            "single mom",
            "her son",
        )
    ):
        relationship = "family"
    elif any(
        word in text
        for word in (
            "married couple",
            "romantic couple",
            "between a couple",
            "are a couple",
            "couple in their",
            "boyfriend",
            "girlfriend",
        )
    ):
        relationship = "romantic_partners"
    elif any(
        word in text
        for word in ("colleague", "co-worker", "meeting", "office", "work", "board members")
    ):
        relationship = "colleagues"
    elif any(word in text for word in ("friend", "neighbors", "roommates", "discussion group")):
        relationship = "friends_or_peers"
    elif any(word in text for word in ("court", "city meeting", "public")):
        relationship = "institutional_or_public"
    else:
        relationship = "unknown"
    confidence = "high" if relationship != "unknown" else "review"
    return relationship, situation, confidence


def resolve_context(
    conversation_id: str, context: str
) -> tuple[str, str, str, str]:
    override = CORE_CONTEXT_OVERRIDES.get(conversation_id.upper())
    if override is not None:
        return (*override, "manual_high", "manual_core_chat_comment_review_v1")
    relationship, situation, confidence = infer_context(context)
    return relationship, situation, confidence, "keyword_rules_v1"


def _profile_from_record(record: MetadataRecord | None) -> dict[str, Any]:
    if record is None:
        return {
            "age_group": "unknown",
            "gender": "unknown",
            "social_role": "unknown",
            "background": "unknown",
        }
    background_parts = [
        f"education={record.education}",
        f"home_state={record.home_state}",
        f"current_state={record.current_state}",
        f"ethnicity={record.ethnicity}",
    ]
    gender = {"f": "female", "female": "female", "m": "male", "male": "male"}.get(
        record.gender, "unknown"
    )
    return {
        "age_group": age_group(record.age),
        "gender": gender,
        "social_role": record.occupation,
        "background": "; ".join(background_parts),
    }


def _record_signature(record: MetadataRecord) -> tuple[str, ...]:
    return (
        record.name,
        record.gender,
        record.age,
        record.hometown,
        record.home_state,
        record.current_state,
        record.education,
        record.years_education,
        record.occupation,
        record.ethnicity,
    )


def _metadata_indices(records: Iterable[MetadataRecord]):
    by_alias: dict[str, list[MetadataRecord]] = defaultdict(list)
    by_key: dict[str, MetadataRecord] = {}
    for record in records:
        by_key[record.record_key] = record
        for alias in metadata_aliases(record.name):
            by_alias[alias].append(record)
    return by_alias, by_key


def match_metadata(
    conversation_id: str,
    speaker_name: str,
    by_alias: dict[str, list[MetadataRecord]],
    by_key: dict[str, MetadataRecord],
) -> tuple[MetadataRecord | None, str]:
    normalized = normalize_speaker_name(speaker_name)
    override = PROFILE_OVERRIDES.get((conversation_id, normalized))
    if override:
        return by_key.get(override), "manual_high_confidence"
    candidates = by_alias.get(normalized, [])
    hinted = [record for record in candidates if record.conversation_hint == conversation_id]
    if len(hinted) == 1:
        return hinted[0], "conversation_hint"
    distinct: dict[tuple[str, ...], MetadataRecord] = {}
    for record in candidates:
        existing = distinct.get(_record_signature(record))
        if existing is None or (not existing.participant_id and record.participant_id):
            distinct[_record_signature(record)] = record
    if len(distinct) == 1:
        return next(iter(distinct.values())), "unique_name"
    if not candidates:
        return None, "not_found"
    return None, "ambiguous_name"


def _speaker_uid(name: str, record: MetadataRecord | None) -> str:
    if record is not None:
        return f"sbcsae:{record.record_key}"
    return f"sbcsae:name:{normalize_speaker_name(name).lower()}"


def _audio_candidate(audio_dir: Path | None, conversation_id: str, media: str) -> str | None:
    if audio_dir is None or not audio_dir.exists():
        return None
    names = (
        f"{conversation_id}.wav",
        f"{media}.wav",
        f"{int(media):02d}.wav" if media.isdigit() else "",
    )
    for name in names:
        if not name:
            continue
        direct = audio_dir / name
        if direct.is_file():
            return str(direct.resolve())
        found = next(audio_dir.rglob(name), None)
        if found is not None:
            return str(found.resolve())
    return None


def _wav_info(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        with wave.open(path, "rb") as wav:
            frame_rate = wav.getframerate()
            frames = wav.getnframes()
            return {
                "channels": wav.getnchannels(),
                "sample_rate_hz": frame_rate,
                "sample_width_bytes": wav.getsampwidth(),
                "frames": frames,
                "duration_s": frames / frame_rate,
            }
    except (EOFError, OSError, wave.Error):
        return None


def _component_groups(conversations: list[dict[str, Any]]) -> None:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for conversation in conversations:
        uids = [p["speaker_uid"] for p in conversation["participants"] if p["is_person"]]
        for uid in uids[1:]:
            union(uids[0], uid)
    for conversation in conversations:
        uids = [p["speaker_uid"] for p in conversation["participants"] if p["is_person"]]
        roots = sorted({find(uid) for uid in uids})
        material = "|".join(roots or [conversation["conversation_id"]])
        digest = hashlib.blake2b(material.encode("utf-8"), digest_size=6).hexdigest()
        conversation["split_group"] = f"speaker-component-{digest}"


def prepare_sbcsae_catalog(
    *,
    trn_dir: str | Path,
    chat_dir: str | Path,
    metadata_dir: str | Path,
    output_dir: str | Path,
    audio_dir: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records, issues = load_metadata(metadata_dir)
    by_alias, by_key = _metadata_indices(records)
    conversations: list[dict[str, Any]] = []
    utterance_rows: list[dict[str, Any]] = []
    profile_statuses: Counter[str] = Counter()

    for trn_path in sorted(Path(trn_dir).glob("SBC*.trn")):
        conversation_id = trn_path.stem.upper()
        chat_path = Path(chat_dir) / f"{conversation_id}.cha"
        if not chat_path.is_file():
            issues.append(
                {"scope": "conversation", "conversation_id": conversation_id, "reason": "missing_chat"}
            )
            continue
        declared, comments, media = parse_chat_headers(chat_path)
        trn_diagnostics: list[dict[str, Any]] = []
        utterances = parse_trn(trn_path, strict=False, diagnostics=trn_diagnostics)
        for diagnostic in trn_diagnostics:
            issues.append(
                {
                    "scope": "trn",
                    "conversation_id": conversation_id,
                    "source_file": trn_path.name,
                    **diagnostic,
                }
            )

        participants = list(declared)
        participant_aliases: dict[str, int] = {}
        for index, participant in enumerate(participants):
            participant_aliases[normalize_speaker_name(participant.code)] = index
            participant_aliases[normalize_speaker_name(participant.display_name)] = index
        observed_names = []
        for utterance in utterances:
            name = normalize_speaker_name(utterance.speaker)
            if name not in observed_names:
                observed_names.append(name)
            if name not in participant_aliases:
                record, _ = match_metadata(conversation_id, name, by_alias, by_key)
                role = "Speaker" if record is not None and is_person_name(name) else "Unmapped"
                participants.append(ChatParticipant(name, name, role))
                participant_aliases[name] = len(participants) - 1
                issues.append(
                    {
                        "scope": "speaker",
                        "conversation_id": conversation_id,
                        "speaker": name,
                        "reason": "undeclared_trn_speaker_recovered"
                        if role == "Speaker"
                        else "undeclared_non_person_label",
                    }
                )

        participant_rows: list[dict[str, Any]] = []
        for index, participant in enumerate(participants):
            name = normalize_speaker_name(participant.display_name)
            person = is_person_name(name, participant.role)
            record, status = (
                match_metadata(conversation_id, name, by_alias, by_key)
                if person
                else (None, "not_applicable")
            )
            profile_statuses[status] += 1
            if person and record is None:
                issues.append(
                    {
                        "scope": "profile",
                        "conversation_id": conversation_id,
                        "speaker": name,
                        "reason": f"metadata_{status}",
                    }
                )
            local_id = f"speaker_{sum(1 for row in participant_rows if row['is_person']):02d}" if person else f"non_person_{index:02d}"
            participant_rows.append(
                {
                    "local_speaker_id": local_id,
                    "speaker_uid": _speaker_uid(name, record),
                    "code": participant.code,
                    "display_name": name,
                    "role": participant.role,
                    "is_person": person,
                    "metadata_match_status": status,
                    "metadata_record_key": record.record_key if record else None,
                    "profile": _profile_from_record(record),
                }
            )
        alias_to_row: dict[str, dict[str, Any]] = {}
        for participant, row in zip(participants, participant_rows):
            alias_to_row[normalize_speaker_name(participant.code)] = row
            alias_to_row[normalize_speaker_name(participant.display_name)] = row

        for utterance_index, utterance in enumerate(utterances):
            speaker_name = normalize_speaker_name(utterance.speaker)
            speaker = alias_to_row[speaker_name]
            utterance_rows.append(
                {
                    "utterance_id": f"{conversation_id}-{utterance_index:06d}",
                    "conversation_id": conversation_id,
                    "start_s": utterance.start_s,
                    "end_s": utterance.end_s,
                    "speaker": speaker["local_speaker_id"],
                    "speaker_uid": speaker["speaker_uid"] if speaker["is_person"] else None,
                    "is_person": speaker["is_person"],
                    "raw_speaker": utterance.speaker,
                    "text": utterance.text,
                    "clean_text": clean_transcript_text(utterance.text),
                }
            )

        context = " ".join(comments)
        relationship, situation, context_confidence, context_method = resolve_context(
            conversation_id, context
        )
        observed_humans = {
            alias_to_row[name]["speaker_uid"]
            for name in observed_names
            if name in alias_to_row and alias_to_row[name]["is_person"]
        }
        observed_unit_counts: Counter[str] = Counter()
        for utterance in utterances:
            row = alias_to_row[normalize_speaker_name(utterance.speaker)]
            if row["is_person"]:
                observed_unit_counts[row["speaker_uid"]] += 1
        human_unit_total = sum(observed_unit_counts.values())
        minimum_human_share = (
            min(observed_unit_counts.values()) / human_unit_total
            if observed_unit_counts and human_unit_total
            else 0.0
        )
        core_dyadic = (
            len(observed_humans) == 2
            and relationship != "speaker_audience"
            and minimum_human_share >= 0.05
        )
        selected_audio = _audio_candidate(
            Path(audio_dir) if audio_dir else None, conversation_id, media
        )
        audio_info = _wav_info(selected_audio)
        if selected_audio is not None and audio_info is None:
            issues.append(
                {
                    "scope": "audio",
                    "conversation_id": conversation_id,
                    "reason": "invalid_wav_header",
                    "audio_path": selected_audio,
                }
            )
        conversations.append(
            {
                "schema_version": "1.0",
                "conversation_id": conversation_id,
                "title": comments[0] if comments else conversation_id,
                "comments": comments,
                "media_id": media,
                "duration_s": max(item.end_s for item in utterances),
                "utterance_count": len(utterances),
                "declared_human_speaker_count": sum(
                    is_person_name(item.display_name, item.role) for item in declared
                ),
                "observed_human_speaker_count": len(observed_humans),
                "declared_dyadic": sum(
                    is_person_name(item.display_name, item.role) for item in declared
                )
                == 2,
                "observed_dyadic": len(observed_humans) == 2,
                "core_dyadic": core_dyadic,
                "minimum_observed_human_unit_share": minimum_human_share,
                "relationship": relationship,
                "situation": situation,
                "context_mapping_method": context_method,
                "context_mapping_confidence": context_confidence,
                "participants": participant_rows,
                "audio_path": selected_audio,
                "audio_info": audio_info,
                "trn_path": str(trn_path.resolve()),
                "chat_path": str(chat_path.resolve()),
            }
        )

    _component_groups(conversations)
    conversation_by_id = {row["conversation_id"]: row for row in conversations}
    for utterance in utterance_rows:
        utterance["split_group"] = conversation_by_id[utterance["conversation_id"]]["split_group"]

    dyadic = [row for row in conversations if row["observed_dyadic"]]
    core_dyadic = [row for row in conversations if row["core_dyadic"]]
    write_jsonl(destination / "conversations.jsonl", conversations)
    write_jsonl(destination / "utterances.jsonl", utterance_rows)
    write_jsonl(destination / "dyadic_conversations.jsonl", dyadic)
    write_jsonl(destination / "core_dyadic_conversations.jsonl", core_dyadic)
    write_jsonl(destination / "issues.jsonl", issues)
    summary = {
        "schema_version": "1.0",
        "conversations": len(conversations),
        "utterances": len(utterance_rows),
        "duration_hours_from_timestamps": sum(row["duration_s"] for row in conversations) / 3600,
        "declared_dyadic_conversations": sum(row["declared_dyadic"] for row in conversations),
        "observed_dyadic_conversations": len(dyadic),
        "core_dyadic_conversations": len(core_dyadic),
        "core_dyadic_definition": (
            "exactly two observed human speakers; neither has <5% of timed human units; "
            "relationship is not speaker_audience"
        ),
        "declared_human_speakers": sum(row["declared_human_speaker_count"] for row in conversations),
        "unique_split_groups": len({row["split_group"] for row in conversations}),
        "audio_files_found": sum(bool(row["audio_path"]) for row in conversations),
        "audio_duration_hours": sum(
            row["audio_info"]["duration_s"] for row in conversations if row["audio_info"]
        )
        / 3600,
        "audio_formats": dict(
            Counter(
                (
                    f"channels={row['audio_info']['channels']};"
                    f"sample_rate_hz={row['audio_info']['sample_rate_hz']};"
                    f"sample_width_bytes={row['audio_info']['sample_width_bytes']}"
                )
                for row in conversations
                if row["audio_info"]
            )
        ),
        "profile_match_statuses": dict(profile_statuses),
        "relationships": dict(Counter(row["relationship"] for row in conversations)),
        "situations": dict(Counter(row["situation"] for row in conversations)),
        "issues": len(issues),
        "issue_reasons": dict(Counter(row["reason"] for row in issues)),
        "outputs": {
            "conversations": str((destination / "conversations.jsonl").resolve()),
            "utterances": str((destination / "utterances.jsonl").resolve()),
            "dyadic_conversations": str((destination / "dyadic_conversations.jsonl").resolve()),
            "core_dyadic_conversations": str(
                (destination / "core_dyadic_conversations.jsonl").resolve()
            ),
            "issues": str((destination / "issues.jsonl").resolve()),
        },
    }
    write_json(destination / "summary.json", summary)
    return summary
