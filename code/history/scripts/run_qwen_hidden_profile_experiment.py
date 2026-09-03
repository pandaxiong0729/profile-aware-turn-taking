from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile_turntaking.qwen_hidden_profile_experiment import (
    CONTEXT_POOLING_MODES,
    DEFAULT_QWEN_MODEL,
    QwenHiddenTrainConfig,
    QwenThinkerHiddenEncoder,
    build_qwen_hidden_cache,
    run_qwen_hidden_profile_experiment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen-Omni Thinker hidden-state profile adapter experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser(
        "cache",
        help="Extract frozen Qwen hidden vectors for one paired request split.",
    )
    cache.add_argument("--run-dir", required=True)
    cache.add_argument("--cache-path", required=True)
    cache.add_argument("--model-name", default=DEFAULT_QWEN_MODEL)
    cache.add_argument("--torch-dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    cache.add_argument("--device-map", default="auto")
    cache.add_argument("--model-cache-dir")
    cache.add_argument("--offload-folder")
    cache.add_argument("--local-files-only", action="store_true")
    cache.add_argument("--limit", type=int)
    cache.add_argument("--sample-rate", type=int, default=16_000)
    cache.add_argument("--encoder-batch-size", type=int, default=1)
    cache.add_argument("--checkpoint-every", type=int, default=100)
    cache.add_argument("--no-resume", action="store_true")
    cache.add_argument(
        "--context-mode",
        choices=("audio_transcript", "audio_only"),
        default="audio_transcript",
    )
    cache.add_argument(
        "--context-pooling",
        choices=CONTEXT_POOLING_MODES,
        default="prompt_last",
    )

    train = subparsers.add_parser(
        "train",
        help="Train/evaluate the small profile adapter over cached Qwen vectors.",
    )
    train.add_argument("--train-cache", required=True)
    train.add_argument("--val-cache", required=True)
    train.add_argument("--test-cache", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 71])
    train.add_argument("--device", default="cpu")
    train.add_argument("--fusion", choices=("gate", "concat"), default="gate")
    train.add_argument("--hidden-dimension", type=int, default=256)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--profile-dropout", type=float, default=0.5)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=8e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)

    full = subparsers.add_parser(
        "full",
        help="Cache train/val/test splits, then train/evaluate.",
    )
    full.add_argument(
        "--data-dir",
        default="data/processed/sbcsae_semantic_profile_v1",
        help="Directory containing train/val/test paired request folders.",
    )
    full.add_argument("--cache-dir", required=True)
    full.add_argument("--output-dir", required=True)
    full.add_argument("--model-name", default=DEFAULT_QWEN_MODEL)
    full.add_argument("--torch-dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    full.add_argument("--device-map", default="auto")
    full.add_argument("--model-cache-dir")
    full.add_argument("--offload-folder")
    full.add_argument("--local-files-only", action="store_true")
    full.add_argument("--limit", type=int, help="Optional per-split limit for smoke runs.")
    full.add_argument("--sample-rate", type=int, default=16_000)
    full.add_argument("--encoder-batch-size", type=int, default=1)
    full.add_argument("--checkpoint-every", type=int, default=100)
    full.add_argument("--no-resume", action="store_true")
    full.add_argument("--cache-only", action="store_true")
    full.add_argument("--overwrite-cache", action="store_true")
    full.add_argument(
        "--context-mode",
        choices=("audio_transcript", "audio_only"),
        default="audio_transcript",
    )
    full.add_argument(
        "--context-pooling",
        choices=CONTEXT_POOLING_MODES,
        default="prompt_last",
    )
    full.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 71])
    full.add_argument("--device", default="cpu")
    full.add_argument("--fusion", choices=("gate", "concat"), default="gate")
    full.add_argument("--hidden-dimension", type=int, default=256)
    full.add_argument("--dropout", type=float, default=0.2)
    full.add_argument("--profile-dropout", type=float, default=0.5)
    full.add_argument("--epochs", type=int, default=40)
    full.add_argument("--patience", type=int, default=8)
    full.add_argument("--batch-size", type=int, default=64)
    full.add_argument("--learning-rate", type=float, default=8e-4)
    full.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def _train_config(args: argparse.Namespace) -> QwenHiddenTrainConfig:
    return QwenHiddenTrainConfig(
        hidden_dimension=args.hidden_dimension,
        dropout=args.dropout,
        profile_dropout=args.profile_dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seeds=tuple(args.seeds),
        device=args.device,
        fusion=args.fusion,
    )


def main() -> None:
    args = _parse_args()
    if args.command == "cache":
        meta = build_qwen_hidden_cache(
            args.run_dir,
            args.cache_path,
            model_name=args.model_name,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            model_cache_dir=args.model_cache_dir,
            offload_folder=args.offload_folder,
            limit=args.limit,
            sample_rate=args.sample_rate,
            context_mode=args.context_mode,
            encoder_batch_size=args.encoder_batch_size,
            checkpoint_every=args.checkpoint_every,
            resume=not args.no_resume,
            context_pooling=args.context_pooling,
        )
        print(meta)
        return

    if args.command == "train":
        summary = run_qwen_hidden_profile_experiment(
            args.train_cache,
            args.val_cache,
            args.test_cache,
            args.output_dir,
            config=_train_config(args),
        )
        print(summary["aggregate"])
        return

    if args.command == "full":
        data_dir = Path(args.data_dir)
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        encoder = QwenThinkerHiddenEncoder(
            args.model_name,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            cache_dir=args.model_cache_dir,
            offload_folder=args.offload_folder,
        )
        caches = {}
        for split in ("train", "val", "test"):
            cache_path = cache_dir / f"{split}.qwen-hidden.npz"
            meta_path = cache_path.with_suffix(".meta.json")
            reuse = False
            if cache_path.is_file() and meta_path.is_file() and not args.overwrite_cache:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                reuse = (
                    str(Path(meta.get("source_run_dir", "")).resolve())
                    == str((data_dir / split).resolve())
                    and str(meta.get("model_name")) == str(args.model_name)
                    and str(meta.get("context_mode")) == str(args.context_mode)
                    and str(meta.get("context_pooling", "prompt_last"))
                    == str(args.context_pooling)
                    and (args.limit is None or int(meta.get("samples", -1)) == int(args.limit))
                )
            if reuse:
                print(f"[qwen-hidden-cache] reuse complete cache {cache_path}", flush=True)
            else:
                build_qwen_hidden_cache(
                    data_dir / split,
                    cache_path,
                    model_name=args.model_name,
                    torch_dtype=args.torch_dtype,
                    device_map=args.device_map,
                    local_files_only=args.local_files_only,
                    model_cache_dir=args.model_cache_dir,
                    offload_folder=args.offload_folder,
                    limit=args.limit,
                    sample_rate=args.sample_rate,
                    encoder=encoder,
                    context_mode=args.context_mode,
                    encoder_batch_size=args.encoder_batch_size,
                    checkpoint_every=args.checkpoint_every,
                    resume=not args.no_resume,
                    context_pooling=args.context_pooling,
                )
            caches[split] = cache_path
        if args.cache_only:
            print({split: str(path.resolve()) for split, path in caches.items()})
            return
        summary = run_qwen_hidden_profile_experiment(
            caches["train"],
            caches["val"],
            caches["test"],
            args.output_dir,
            config=_train_config(args),
        )
        print(summary["aggregate"])
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
