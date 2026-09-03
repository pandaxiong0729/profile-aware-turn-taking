from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from profile_turntaking.utils import read_jsonl, write_json, write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_asr_to_prompt(prompt: str, transcript: str) -> str:
    marker = "\nProfile condition:\n"
    if marker not in prompt:
        raise ValueError("Prompt does not contain the profile marker")
    asr_section = "\n".join(
        [
            "",
            "Causal ASR transcript generated from this exact audio:",
            transcript or "(No intelligible speech was transcribed.)",
            "This transcript also stops at t; never continue it beyond the audio.",
            "",
        ]
    )
    return prompt.replace(marker, asr_section + marker, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the same hidden benchmark with causal Qwen ASR added to each prompt."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_root = Path(args.source_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    source_requests = list(read_jsonl(source_root / "requests.jsonl"))
    references = list(read_jsonl(source_root / "reference_labels.jsonl"))
    asr_rows = {
        str(row["sample_id"]): row for row in read_jsonl(source_root / "asr.jsonl")
    }
    output_audio = output_root / "causal_audio"
    output_audio.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    requests: list[dict[str, Any]] = []
    for source in source_requests:
        sample_id = str(source["sample_id"])
        asr = asr_rows.get(sample_id)
        if asr is None:
            errors.append(f"{sample_id}: missing ASR")
            continue
        if asr.get("error"):
            errors.append(f"{sample_id}: ASR error: {asr['error']}")
            continue
        if asr["audio_sha256"] != source["audio_sha256"]:
            errors.append(f"{sample_id}: ASR audio hash differs from source request")
            continue

        source_audio = Path(str(source["audio_path"]))
        if not source_audio.is_absolute():
            source_audio = source_root / source_audio
        target_audio = output_audio / f"{sample_id}.wav"
        if not target_audio.exists():
            shutil.copy2(source_audio, target_audio)
        if sha256_file(target_audio) != source["audio_sha256"]:
            errors.append(f"{sample_id}: copied audio hash mismatch")
            continue

        qwen_asr = str(asr["transcript"]).strip()
        prompt = add_asr_to_prompt(str(source["prompt"]), qwen_asr)
        request = dict(source)
        request["request_id"] = f"{sample_id}::hidden-qwen-asr"
        request["audio_path"] = str(target_audio.relative_to(output_root))
        request["qwen_causal_asr"] = qwen_asr
        request["qwen_causal_asr_sha256"] = sha256_text(qwen_asr)
        request["prompt"] = prompt
        request["prompt_sha256"] = sha256_text(prompt)
        requests.append(request)

    reference_by_id = {str(row["sample_id"]): row for row in references}
    if {str(row["sample_id"]) for row in requests} != set(reference_by_id):
        errors.append("request and reference sample IDs differ")

    source_by_id = {str(row["sample_id"]): row for row in source_requests}
    for request in requests:
        sample_id = str(request["sample_id"])
        source = source_by_id[sample_id]
        for field in (
            "sample_id",
            "conversation_id",
            "task",
            "audio_sha256",
            "prediction_boundary_in_conversation_s",
            "profile_mode",
            "profile_text",
            "causal_partial_transcript",
            "transcript_sha256",
            "allowed_predictions",
        ):
            if request[field] != source[field]:
                errors.append(f"{sample_id}: changed invariant field {field}")
        if sha256_text(str(request["causal_partial_transcript"])) != request[
            "transcript_sha256"
        ]:
            errors.append(f"{sample_id}: timed transcript hash mismatch")
        if sha256_text(str(request["qwen_causal_asr"])) != request[
            "qwen_causal_asr_sha256"
        ]:
            errors.append(f"{sample_id}: Qwen ASR hash mismatch")
        if any(
            field in request
            for field in ("target", "reference_label", "label_evidence")
        ):
            errors.append(f"{sample_id}: target or annotation evidence leaked")

    audit = {
        "passed": not errors and len(requests) == len(source_requests),
        "requests": len(requests),
        "source_requests": len(source_requests),
        "task_counts": dict(Counter(row["task"] for row in requests)),
        "same_sample_ids": {
            row["sample_id"] for row in requests
        }
        == {row["sample_id"] for row in source_requests},
        "same_audio_sha256": all(
            row["audio_sha256"] == source_by_id[str(row["sample_id"])]["audio_sha256"]
            for row in requests
        ),
        "same_timed_transcript_sha256": all(
            row["transcript_sha256"]
            == source_by_id[str(row["sample_id"])]["transcript_sha256"]
            for row in requests
        ),
        "same_prediction_boundary": all(
            row["prediction_boundary_in_conversation_s"]
            == source_by_id[str(row["sample_id"])][
                "prediction_boundary_in_conversation_s"
            ]
            for row in requests
        ),
        "same_profile_condition": all(
            row["profile_text"] == source_by_id[str(row["sample_id"])]["profile_text"]
            for row in requests
        ),
        "target_fields_absent_from_requests": all(
            field not in row
            for row in requests
            for field in ("target", "reference_label", "label_evidence")
        ),
        "only_intended_input_change": "added qwen_causal_asr and its prompt section",
        "message_content_order": ["text", "input_audio"],
        "errors": errors,
    }
    write_jsonl(output_root / "requests.jsonl", requests)
    write_jsonl(output_root / "reference_labels.jsonl", references)
    write_json(output_root / "input_audit.json", audit)
    write_json(
        output_root / "run_config.json",
        {
            "diagnostic_only": True,
            "source_dir": str(source_root),
            "model": "Qwen2.5-Omni-3B-Q8_0",
            "input_contract": (
                "same causal mono audio + same completed-unit timed transcript + "
                "Qwen causal ASR from exact audio + same hidden profile"
            ),
            "comparison_constraint": (
                "Only the Qwen causal ASR section is added relative to source_dir."
            ),
            "reference_warning": (
                "Automatic structural weak labels; not human gold labels."
            ),
        },
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise ValueError("Input audit failed: " + "; ".join(errors[:10]))


if __name__ == "__main__":
    main()
