"""Build Talking-Turns-style Qwen audio-layer caches for train/val/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile_turntaking.qwen_audio_layer_experiment import (
    QwenAudioLayerBoundaryEncoder,
    build_qwen_audio_layer_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    base_cache_dir = Path(args.base_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = QwenAudioLayerBoundaryEncoder(
        args.model_name,
        torch_dtype=args.torch_dtype,
        device=args.device,
    )
    summaries = {}
    for split in ("train", "val", "test"):
        summaries[split] = build_qwen_audio_layer_cache(
            data_dir / split,
            base_cache_dir / f"{split}.qwen-hidden.npz",
            output_dir / f"{split}.qwen-hidden.npz",
            model_name=args.model_name,
            torch_dtype=args.torch_dtype,
            device=args.device,
            encoder_batch_size=args.encoder_batch_size,
            checkpoint_every=args.checkpoint_every,
            resume=not args.no_resume,
            limit=args.limit,
            encoder=encoder,
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
