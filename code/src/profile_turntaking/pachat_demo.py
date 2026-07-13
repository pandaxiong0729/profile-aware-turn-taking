"""Normalize the official PAChat project-page demos without overstating their scope."""

from __future__ import annotations

import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from .utils import write_json, write_jsonl

_SPACE_RE = re.compile(r"\s+")
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}


@dataclass
class HtmlNode:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[HtmlNode | str] = field(default_factory=list)


class _TreeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = HtmlNode("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = HtmlNode(tag, {name: value or "" for name, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].children.append(data)


def _nodes(node: HtmlNode, *, tag: str | None = None) -> Iterable[HtmlNode]:
    for child in node.children:
        if not isinstance(child, HtmlNode):
            continue
        if tag is None or child.tag == tag:
            yield child
        yield from _nodes(child, tag=tag)


def _classes(node: HtmlNode) -> set[str]:
    return set(node.attrs.get("class", "").split())


def _find_class(node: HtmlNode, class_name: str) -> list[HtmlNode]:
    return [candidate for candidate in _nodes(node) if class_name in _classes(candidate)]


def _text(node: HtmlNode) -> str:
    parts: list[str] = []

    def visit(current: HtmlNode) -> None:
        for child in current.children:
            if isinstance(child, str):
                parts.append(child)
            else:
                visit(child)

    visit(node)
    return _SPACE_RE.sub(" ", " ".join(parts)).strip()


def _labeled_span(case: HtmlNode, label: str) -> str:
    for span in _nodes(case, tag="span"):
        value = _text(span)
        match = re.search(rf"{label}\s*[:：]\s*(.+)$", value, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "unknown"


def _age_group(description: str) -> str:
    match = re.search(r"\b(\d{1,3})[- ]year[- ]old\b", description, flags=re.IGNORECASE)
    if not match:
        return "unknown"
    age = int(match.group(1))
    if age < 13:
        return "0-12"
    if age < 18:
        return "13-17"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


def _wav_info(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                return None
            format_fields: tuple[int, int, int, int, int, int] | None = None
            data_bytes: int | None = None
            fact_frames: int | None = None
            while True:
                chunk_header = stream.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
                payload_start = stream.tell()
                if chunk_id == b"fmt " and chunk_size >= 16:
                    format_fields = struct.unpack("<HHIIHH", stream.read(16))
                elif chunk_id == b"fact" and chunk_size >= 4:
                    fact_frames = struct.unpack("<I", stream.read(4))[0]
                elif chunk_id == b"data":
                    data_bytes = chunk_size
                stream.seek(payload_start + chunk_size + (chunk_size % 2))
        if format_fields is None or data_bytes is None:
            return None
        format_tag, channels, sample_rate, byte_rate, block_align, bits = format_fields
        frames = fact_frames or (data_bytes // block_align if block_align else 0)
        duration = data_bytes / byte_rate if byte_rate else frames / sample_rate
        return {
            "encoding": {1: "pcm_integer", 3: "ieee_float"}.get(
                format_tag, f"wav_format_{format_tag}"
            ),
            "format_tag": format_tag,
            "channels": channels,
            "sample_rate_hz": sample_rate,
            "sample_width_bytes": bits // 8,
            "frames": frames,
            "duration_s": duration,
        }
    except (OSError, struct.error):
        return None


def _profile_rows(case: HtmlNode, case_id: str) -> list[dict[str, Any]]:
    profile_containers = [
        node
        for node in _nodes(case, tag="div")
        if "background: #f5f5f5" in node.attrs.get("style", "")
    ]
    if not profile_containers:
        return []
    profiles: list[dict[str, Any]] = []
    for child in profile_containers[0].children:
        if not isinstance(child, HtmlNode) or child.tag != "div":
            continue
        spans = list(_nodes(child, tag="span"))
        if len(spans) < 3:
            continue
        name = _text(spans[0])
        role = _text(spans[1]).strip("() ") or "unknown"
        sentences = [_text(span) for span in spans[2:] if _text(span)]
        description = " ".join(sentences)
        profiles.append(
            {
                "profile_id": f"{case_id}-profile-{len(profiles) + 1:02d}",
                "case_id": case_id,
                "name": name,
                "short_name": name.split()[0] if name else "unknown",
                "role": role,
                "natural_language_profile": sentences,
                "structured_profile": {
                    "age_group": _age_group(description),
                    "gender": "unknown",
                    "social_role": role,
                    "background": description or "unknown",
                },
                "source": "official_project_page_demo",
            }
        )
    return profiles


def parse_pachat_demo(site_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(site_dir)
    parser = _TreeParser()
    parser.feed((root / "index.html").read_text(encoding="utf-8", errors="replace"))
    cases: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_text: dict[str, str] = {}

    for case_index, case_node in enumerate(_find_class(parser.root, "case-card"), 1):
        headings = list(_nodes(case_node, tag="h2"))
        case_name = _text(headings[0]) if headings else f"Case {case_index}"
        case_id = f"pachat-demo-{case_index:02d}"
        case_profiles = _profile_rows(case_node, case_id)
        profiles.extend(case_profiles)
        profile_by_alias: dict[str, str] = {}
        for row in case_profiles:
            name_parts = row["name"].lower().split()
            for alias in {row["name"].lower(), name_parts[0], name_parts[-1]}:
                profile_by_alias[alias] = row["profile_id"]
        case_turns = _find_class(case_node, "dialogue-case")
        turn_ids: list[str] = []
        for turn_index, turn_node in enumerate(case_turns, 1):
            speakers = _find_class(turn_node, "speaker-name")
            texts = _find_class(turn_node, "text")
            sources = [node for node in _nodes(turn_node, tag="source") if node.attrs.get("src")]
            raw_speaker = _text(speakers[0]) if speakers else "unknown"
            system_turn = "right" in _classes(turn_node)
            speaker = "assistant" if system_turn else raw_speaker
            transcript = _text(texts[0]) if texts else ""
            relative_audio = sources[0].attrs["src"] if sources else ""
            audio_path = root / relative_audio if relative_audio else None
            info = _wav_info(audio_path) if audio_path and audio_path.is_file() else None
            turn_id = f"{case_id}-turn-{turn_index:02d}"
            turn_ids.append(turn_id)
            turns.append(
                {
                    "turn_id": turn_id,
                    "case_id": case_id,
                    "turn_index": turn_index,
                    "speaker": speaker,
                    "speaker_raw": raw_speaker,
                    "speaker_type": "assistant" if system_turn else "user",
                    "profile_id": None
                    if system_turn
                    else profile_by_alias.get(raw_speaker.lower()),
                    "text": transcript,
                    "audio_path": str(audio_path.resolve()) if audio_path else None,
                    "audio_relative_path": relative_audio or None,
                    "audio_info": info,
                    "continuous_timing_available": False,
                    "turntaking_label_eligible": False,
                }
            )
            if not system_turn and raw_speaker.lower() not in profile_by_alias:
                issues.append(
                    {
                        "scope": "profile_mapping",
                        "case_id": case_id,
                        "turn_id": turn_id,
                        "reason": "speaker_profile_not_found",
                        "speaker": raw_speaker,
                    }
                )
            if info is None:
                issues.append(
                    {
                        "scope": "audio",
                        "case_id": case_id,
                        "turn_id": turn_id,
                        "reason": "missing_or_invalid_audio",
                        "audio_relative_path": relative_audio or None,
                    }
                )
            normalized_text = _SPACE_RE.sub(" ", transcript.lower()).strip()
            if normalized_text and normalized_text in seen_text:
                issues.append(
                    {
                        "scope": "content",
                        "case_id": case_id,
                        "turn_id": turn_id,
                        "reason": "exact_duplicate_transcript_across_demo_turns",
                        "duplicate_of": seen_text[normalized_text],
                    }
                )
            elif normalized_text:
                seen_text[normalized_text] = turn_id
        cases.append(
            {
                "schema_version": "1.0",
                "case_id": case_id,
                "source_case_name": case_name,
                "scenario": _labeled_span(case_node, "Scenario"),
                "theme": _labeled_span(case_node, "Theme"),
                "profile_ids": [row["profile_id"] for row in case_profiles],
                "turn_ids": turn_ids,
                "dataset_scope": "official_project_page_demo_only",
                "continuous_timing_available": False,
                "turntaking_label_eligible": False,
            }
        )
    return cases, profiles, turns, issues


def prepare_pachat_demo(*, site_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cases, profiles, turns, issues = parse_pachat_demo(site_dir)
    issues[:0] = [
        {
            "scope": "release",
            "reason": "full_persona_dialogue_release_not_found",
            "detail": "Only official project-page demos were available at preprocessing time.",
        },
        {
            "scope": "license",
            "reason": "license_not_specified_in_official_demo_repository",
            "detail": "Do not redistribute the demo archive until the publisher clarifies its license.",
        },
        {
            "scope": "task_fit",
            "reason": "isolated_turn_audio_has_no_continuous_turn_timing",
            "detail": "The demos cannot provide gap, overlap, or 40 ms turn-taking gold labels.",
        },
    ]
    write_jsonl(destination / "cases.jsonl", cases)
    write_jsonl(destination / "profiles.jsonl", profiles)
    write_jsonl(destination / "turns.jsonl", turns)
    write_jsonl(destination / "issues.jsonl", issues)
    audio_rows = [row for row in turns if row["audio_info"]]
    summary = {
        "schema_version": "1.0",
        "dataset_name": "Persona-Dialogue",
        "model_name": "PAChat",
        "dataset_scope": "official_project_page_demo_only",
        "full_release_available": False,
        "license_status": "not_specified_in_official_demo_repository",
        "turntaking_label_eligible": False,
        "cases": len(cases),
        "profiles": len(profiles),
        "turns": len(turns),
        "audio_files_valid": len(audio_rows),
        "audio_duration_minutes": sum(row["audio_info"]["duration_s"] for row in audio_rows) / 60,
        "speakers": dict(Counter(row["speaker_type"] for row in turns)),
        "scenarios": dict(Counter(row["scenario"] for row in cases)),
        "issues": len(issues),
        "issue_reasons": dict(Counter(row["reason"] for row in issues)),
        "paper_reported_full_dataset": {
            "dialogues": 21760,
            "turns": 159933,
            "hours": 217,
            "scenarios": 21,
            "source": "Fu et al. (EMNLP 2025)",
        },
        "outputs": {
            "cases": str((destination / "cases.jsonl").resolve()),
            "profiles": str((destination / "profiles.jsonl").resolve()),
            "turns": str((destination / "turns.jsonl").resolve()),
            "issues": str((destination / "issues.jsonl").resolve()),
        },
    }
    write_json(destination / "summary.json", summary)
    return summary
