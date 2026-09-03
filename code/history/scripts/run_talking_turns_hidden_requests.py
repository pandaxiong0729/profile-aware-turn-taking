from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from profile_turntaking.utils import read_jsonl, write_json


def parse_prediction(raw: str, allowed: list[str]) -> str | None:
    upper = raw.upper()
    matches = [
        label
        for label in sorted(allowed, key=len, reverse=True)
        if re.search(rf"\b{re.escape(label)}\b", upper)
    ]
    return matches[0] if matches else None


def run(args: argparse.Namespace) -> None:
    root = Path(args.run_dir).resolve()
    requests = list(read_jsonl(root / "requests.jsonl"))
    output_path = root / "responses.jsonl"
    existing = list(read_jsonl(output_path)) if output_path.exists() else []
    completed = {str(row["request_id"]) for row in existing}
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(requests, start=1):
            if str(row["request_id"]) in completed:
                continue
            audio_path = Path(str(row["audio_path"]))
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
            body = json.dumps(
                {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": row["prompt"]},
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": audio_data,
                                        "format": "wav",
                                    },
                                },
                            ],
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 48,
                    "seed": args.seed,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            started = time.perf_counter()
            raw = ""
            error = ""
            try:
                request = urllib.request.Request(
                    args.endpoint,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=args.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict)
                    )
                raw = str(content)
            except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            prediction = (
                parse_prediction(raw, list(row["allowed_predictions"]))
                if not error
                else None
            )
            result = {
                "request_id": row["request_id"],
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "task": row["task"],
                "profile_mode": row["profile_mode"],
                "model": args.model,
                "audio_sha256": row["audio_sha256"],
                "transcript_sha256": row["transcript_sha256"],
                "prompt_sha256": row["prompt_sha256"],
                "prediction": prediction,
                "valid": prediction in row["allowed_predictions"],
                "raw_response": raw,
                "error": error,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "n": index,
                        "task": row["task"],
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


def score(args: argparse.Namespace) -> None:
    root = Path(args.run_dir).resolve()
    references = {
        str(row["sample_id"]): row for row in read_jsonl(root / "reference_labels.jsonl")
    }
    responses = list(read_jsonl(root / "responses.jsonl"))
    rows: list[dict[str, Any]] = []
    for response in responses:
        reference = references[str(response["sample_id"])]
        prediction = response.get("prediction")
        rows.append(
            {
                "sample_id": response["sample_id"],
                "conversation_id": response["conversation_id"],
                "task": response["task"],
                "prediction": prediction,
                "weak_reference": reference["target"],
                "valid": bool(response.get("valid")),
                "correct": prediction == reference["target"],
            }
        )
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(values)
        distribution = Counter(str(row["prediction"]) for row in values)
        dominant = max(distribution.values(), default=0)
        return {
            "samples": total,
            "valid": sum(row["valid"] for row in values),
            "correct": sum(row["correct"] for row in values),
            "accuracy_against_weak_reference": (
                sum(row["correct"] for row in values) / total if total else None
            ),
            "prediction_distribution": dict(distribution),
            "distinct_predictions": len(distribution),
            "dominant_fraction": dominant / total if total else None,
            "noncollapsed": len(distribution) >= 2 and dominant / total <= 0.9,
        }

    report = {
        "diagnostic_only": True,
        "reference_warning": "Automatic structural weak labels; not human gold labels.",
        "overall": summarize(rows),
        "by_task": {
            task: summarize(values) for task, values in sorted(by_task.items())
        },
        "hidden_gate": {
            "all_tasks_noncollapsed": bool(by_task)
            and all(summarize(values)["noncollapsed"] for values in by_task.values()),
            "profile_effect_claim_allowed": False,
            "reason": (
                "Audio sensitivity must still pass before any profile comparison."
                if by_task
                and all(summarize(values)["noncollapsed"] for values in by_task.values())
                else "At least one hidden task is collapsed; do not run or interpret profile comparisons."
            ),
        },
        "rows": rows,
    }
    write_json(root / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or score strict-causal hidden turn-taking requests."
    )
    parser.add_argument("command", choices=("run", "score"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen2.5-Omni-3B-Q8_0")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    run(args) if args.command == "run" else score(args)


if __name__ == "__main__":
    main()
