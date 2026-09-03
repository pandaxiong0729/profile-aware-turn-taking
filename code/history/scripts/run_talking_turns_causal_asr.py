from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from profile_turntaking.utils import read_jsonl


ASR_PROMPT = """Transcribe the attached English conversation.
The audio ends exactly at the current prediction time.
Write every intelligible word that is actually audible, including incomplete words
at the very end. Do not guess, complete, or predict anything after the audio ends.
Output only the transcript, with no explanation."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def response_text(payload: dict[str, object]) -> str:
    choices = payload["choices"]
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0]["message"]
    content = message["content"]
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        ).strip()
    return str(content).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate causal ASR for the exact audio used by a turn-taking run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen2.5-Omni-3B-Q8_0")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    root = Path(args.run_dir).resolve()
    all_requests = list(read_jsonl(root / "requests.jsonl"))
    requests: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in all_requests:
        key = (str(row["sample_id"]), str(row["audio_sha256"]))
        if key in seen:
            continue
        seen.add(key)
        requests.append(row)
    if args.limit is not None:
        requests = requests[: args.limit]
    output_path = root / "asr.jsonl"
    existing = list(read_jsonl(output_path)) if output_path.exists() else []
    completed = {
        (str(row["sample_id"]), str(row["audio_sha256"]))
        for row in existing
        if not row.get("error")
    }

    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(requests, start=1):
            key = (str(row["sample_id"]), str(row["audio_sha256"]))
            if key in completed:
                continue
            audio_path = Path(str(row["audio_path"]))
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
            request_body = json.dumps(
                {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ASR_PROMPT},
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
                    "max_tokens": 768,
                    "seed": args.seed,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            started = time.perf_counter()
            transcript = ""
            error = ""
            try:
                request = urllib.request.Request(
                    args.endpoint,
                    data=request_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(
                    request, timeout=args.timeout_s
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                transcript = response_text(payload)
            except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            result = {
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "audio_sha256": row["audio_sha256"],
                "prediction_boundary_in_conversation_s": row.get(
                    "prediction_boundary_in_conversation_s",
                    row.get("decision_time_in_conversation_s"),
                ),
                "model": args.model,
                "prompt": ASR_PROMPT,
                "prompt_sha256": sha256_text(ASR_PROMPT),
                "transcript": transcript,
                "transcript_sha256": sha256_text(transcript),
                "error": error,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if not error:
                completed.add(key)
            print(
                json.dumps(
                    {
                        "n": index,
                        "sample_id": row["sample_id"],
                        "characters": len(transcript),
                        "error": error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
