"""Copy a Qwen event-eval run directory and rewrite only the prompt style.

The script preserves the experiment input contract:

- same sample IDs;
- same causal audio files and hashes;
- same causal transcript;
- same prediction boundary;
- same hidden/given/shuffled profile text;
- no reference labels are copied into requests.

Only the rendered prompt text and its derived hashes change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from profile_turntaking.qwen25_omni_event_eval import build_event_prompt


PROFILE_PLACEHOLDER = "<PROFILE_CONDITION>"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_template(prompt: str, profile_text: str) -> str:
    if prompt.count(profile_text) != 1:
        raise ValueError("profile text does not occur exactly once in prompt")
    return prompt.replace(profile_text, PROFILE_PLACEHOLDER, 1)


def rewrite_prompt_style(source_dir: str | Path, output_dir: str | Path, *, prompt_style: str) -> dict[str, Any]:
    source = Path(source_dir).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    for name in ("reference_labels.jsonl", "run_config.json"):
        shutil.copy2(source / name, destination / name)
    if (source / "input_audit.json").is_file():
        shutil.copy2(source / "input_audit.json", destination / "source_input_audit.json")

    audio_source = source / "causal_audio"
    audio_destination = destination / "causal_audio"
    audio_destination.mkdir(parents=True, exist_ok=True)
    for wav in audio_source.glob("*.wav"):
        target = audio_destination / wav.name
        if not target.exists() or target.stat().st_size != wav.stat().st_size:
            shutil.copy2(wav, target)

    requests = []
    for row in _read_jsonl(source / "requests.jsonl"):
        prompt = build_event_prompt(
            audio_duration_s=float(row["audio_duration_s"]),
            transcript=str(row.get("transcript_prefix", "")),
            profile_text=str(row["profile_text"]),
            boundary_state_text=str(row.get("boundary_state_text", "")),
            causal_asr_transcript=str(row.get("causal_asr_transcript", "")),
            forecast_offset_ms=int(row["forecast_offset_ms"]),
            horizon_ms=int(row["horizon_ms"]),
            prompt_style=prompt_style,
        )
        new_row = dict(row)
        new_row["prompt"] = prompt
        new_row["prompt_template_sha256"] = _sha256_text(
            _prompt_template(prompt, str(row["profile_text"]))
        )
        new_row["request_sha256"] = _sha256_text(
            str(row["audio_sha256"])
            + "\n"
            + str(row.get("transcript_sha256", ""))
            + "\n"
            + str(row.get("boundary_state_sha256", ""))
            + "\n"
            + str(row.get("causal_asr_sha256", ""))
            + "\n"
            + prompt
        )
        requests.append(new_row)
    _write_jsonl(destination / "requests.jsonl", requests)

    config = json.loads((destination / "run_config.json").read_text(encoding="utf-8"))
    config["prompt_style"] = prompt_style
    config["prompt_rewritten_from"] = str(source)
    config["prompt_rewrite_only_changed"] = [
        "prompt",
        "prompt_template_sha256",
        "request_sha256",
        "run_config.prompt_style",
    ]
    (destination / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "source_dir": str(source),
        "output_dir": str(destination),
        "prompt_style": prompt_style,
        "requests": len(requests),
        "audio_files": len(list(audio_destination.glob("*.wav"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompt-style",
        required=True,
        choices=("direct", "decision_tree", "reasoned", "hierarchical"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            rewrite_prompt_style(args.source_dir, args.output_dir, prompt_style=args.prompt_style),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
