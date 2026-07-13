"""Command-line interface for local smoke tests and cloud experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import merge_manifests, prepare_sbcsae
from .evaluation import evaluate_checkpoint
from .model import ModelConfig
from .pachat_demo import prepare_pachat_demo
from .quality import audit_preprocessed_data
from .sbcsae_corpus import prepare_sbcsae_catalog
from .sbcsae_manifest import prepare_sbcsae_manifests
from .training import TrainConfig, train_model


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _prepare_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_sbcsae(
        trn_path=args.trn,
        profile_path=args.profile,
        output_manifest=args.manifest,
        audio_path=args.audio,
        synthetic_audio_path=args.synthetic_audio,
        conversation_id=args.conversation_id,
        split_group=args.split_group,
        context_seconds=args.context_seconds,
        horizon_ms=args.horizon_ms,
        stride_ms=args.stride_ms,
        max_per_class=args.max_per_class,
        max_time_s=args.max_time_s,
        seed=args.seed,
    )


def _command_prepare(args: argparse.Namespace) -> None:
    _print(_prepare_from_args(args))


def _command_prepare_corpus(args: argparse.Namespace) -> None:
    _print(
        prepare_sbcsae_catalog(
            trn_dir=args.trn_dir,
            chat_dir=args.chat_dir,
            metadata_dir=args.metadata_dir,
            output_dir=args.output_dir,
            audio_dir=args.audio_dir,
        )
    )


def _command_prepare_pachat_demo(args: argparse.Namespace) -> None:
    _print(prepare_pachat_demo(site_dir=args.site_dir, output_dir=args.output_dir))


def _command_prepare_manifests(args: argparse.Namespace) -> None:
    _print(
        prepare_sbcsae_manifests(
            catalog_dir=args.catalog_dir,
            output_dir=args.output_dir,
            context_seconds=args.context_seconds,
            horizon_ms=args.horizon_ms,
            frame_stride_ms=args.frame_stride_ms,
            evaluation_stride_ms=args.evaluation_stride_ms,
            max_train_per_class=args.max_train_per_class,
            max_evaluation_per_class=args.max_evaluation_per_class,
            seed=args.seed,
            include_non_core_dyadic=args.include_non_core_dyadic,
        )
    )


def _command_audit(args: argparse.Namespace) -> None:
    _print(
        audit_preprocessed_data(
            sbcsae_catalog_dir=args.sbcsae_catalog_dir,
            sbcsae_manifest=args.sbcsae_manifest,
            pachat_demo_dir=args.pachat_demo_dir,
            output_path=args.output,
        )
    )


def _command_train(args: argparse.Namespace) -> None:
    payload = _load_config(args.config)
    report = train_model(
        args.manifest,
        args.checkpoint,
        model_config=ModelConfig.from_dict(payload.get("model", {})),
        train_config=TrainConfig.from_dict(payload.get("train", {})),
    )
    _print(report)


def _command_merge(args: argparse.Namespace) -> None:
    _print(merge_manifests(args.inputs, args.output, seed=args.seed))


def _command_evaluate(args: argparse.Namespace) -> None:
    reports = evaluate_checkpoint(
        args.manifest,
        args.checkpoint,
        args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
    _print(reports)


def _command_smoke(args: argparse.Namespace) -> None:
    root = Path.cwd()
    code_root = Path(__file__).resolve().parents[2]
    repository_root = code_root.parent
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    local_trn_candidates = (
        root / "data" / "sbcsae" / "openslr" / "TRN" / "SBC041.trn",
        repository_root / "data" / "sbcsae" / "openslr" / "TRN" / "SBC041.trn",
        root / "data" / "sbcsae" / "transcripts_trn" / "TRN" / "SBC041.trn",
        repository_root / "data" / "sbcsae" / "transcripts_trn" / "TRN" / "SBC041.trn",
    )
    local_trn = next(
        (path for path in local_trn_candidates if path.is_file()), local_trn_candidates[0]
    )
    local_profile = (
        repository_root / "intro" / "sbcsae_profile_turntaking_training_example.json"
    )
    use_local = local_trn.is_file() and local_profile.is_file() and not args.bundled_fixture
    trn = local_trn if use_local else code_root / "examples" / "smoke.trn"
    profile = local_profile if use_local else code_root / "examples" / "smoke_profile.json"
    manifest = work / "samples.jsonl"
    data_summary = prepare_sbcsae(
        trn_path=trn,
        profile_path=profile,
        output_manifest=manifest,
        synthetic_audio_path=work / "smoke.wav",
        conversation_id="SBC041-smoke" if use_local else "synthetic-smoke",
        split_group="SBC041-smoke" if use_local else "synthetic-smoke",
        context_seconds=args.context_seconds,
        horizon_ms=40,
        stride_ms=40,
        max_per_class=args.max_per_class,
        max_time_s=args.max_time_s if use_local else 20.0,
        seed=args.seed,
    )
    payload = _load_config(args.config)
    checkpoint = work / "model.pt"
    train_report = train_model(
        str(manifest),
        str(checkpoint),
        model_config=ModelConfig.from_dict(payload.get("model", {})),
        train_config=TrainConfig.from_dict(payload.get("train", {})),
    )
    reports = evaluate_checkpoint(
        str(manifest), str(checkpoint), str(work / "evaluation"), split="test"
    )
    _print(
        {
            "status": "ok",
            "fixture": "local_SBC041_timestamps" if use_local else "bundled_synthetic",
            "data": data_summary,
            "training": train_report,
            "evaluation": reports,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="profile-turntaking")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-sbcsae", help="build a five-class JSONL manifest")
    prepare.add_argument("--trn", required=True)
    prepare.add_argument("--profile", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--audio")
    prepare.add_argument("--synthetic-audio")
    prepare.add_argument("--conversation-id")
    prepare.add_argument("--split-group", help="speaker-connected component id for leakage-safe splitting")
    prepare.add_argument("--context-seconds", type=float, default=30.0)
    prepare.add_argument("--horizon-ms", type=int, default=40)
    prepare.add_argument("--stride-ms", type=int, default=40)
    prepare.add_argument("--max-per-class", type=int, default=2000)
    prepare.add_argument("--max-time-s", type=float)
    prepare.add_argument("--seed", type=int, default=13)
    prepare.set_defaults(func=_command_prepare)

    corpus = subparsers.add_parser(
        "prepare-sbcsae-corpus", help="normalize all SBCSAE transcripts, profiles, and issues"
    )
    corpus.add_argument("--trn-dir", required=True)
    corpus.add_argument("--chat-dir", required=True)
    corpus.add_argument("--metadata-dir", required=True)
    corpus.add_argument("--output-dir", required=True)
    corpus.add_argument("--audio-dir")
    corpus.set_defaults(func=_command_prepare_corpus)

    pachat = subparsers.add_parser(
        "prepare-pachat-demo", help="normalize the official project-page Persona-Dialogue demos"
    )
    pachat.add_argument("--site-dir", required=True)
    pachat.add_argument("--output-dir", required=True)
    pachat.set_defaults(func=_command_prepare_pachat_demo)

    manifests = subparsers.add_parser(
        "prepare-sbcsae-manifests",
        help="build leakage-safe weak-label training and natural-evaluation samples",
    )
    manifests.add_argument("--catalog-dir", required=True)
    manifests.add_argument("--output-dir", required=True)
    manifests.add_argument("--context-seconds", type=float, default=30.0)
    manifests.add_argument("--horizon-ms", type=int, default=40)
    manifests.add_argument("--frame-stride-ms", type=int, default=40)
    manifests.add_argument("--evaluation-stride-ms", type=int, default=200)
    manifests.add_argument("--max-train-per-class", type=int, default=10000)
    manifests.add_argument("--max-evaluation-per-class", type=int, default=5000)
    manifests.add_argument("--seed", type=int, default=13)
    manifests.add_argument("--include-non-core-dyadic", action="store_true")
    manifests.set_defaults(func=_command_prepare_manifests)

    audit = subparsers.add_parser(
        "audit-preprocessed", help="cross-check audio, profiles, labels, and split leakage"
    )
    audit.add_argument("--sbcsae-catalog-dir", required=True)
    audit.add_argument("--sbcsae-manifest", required=True)
    audit.add_argument("--pachat-demo-dir", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=_command_audit)

    merge = subparsers.add_parser("merge-manifests", help="merge manifests and re-split by group")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--seed", type=int, default=13)
    merge.set_defaults(func=_command_merge)

    train = subparsers.add_parser("train", help="train a profile-conditioned checkpoint")
    train.add_argument("--manifest", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--config", default="configs/smoke.json")
    train.set_defaults(func=_command_train)

    evaluate = subparsers.add_parser("evaluate", help="run hidden/given/shuffled evaluation")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(func=_command_evaluate)

    smoke = subparsers.add_parser("smoke", help="run prepare, train, and evaluate end to end")
    smoke.add_argument("--work-dir", default="artifacts/smoke")
    smoke.add_argument("--config", default="configs/smoke.json")
    smoke.add_argument("--context-seconds", type=float, default=3.0)
    smoke.add_argument("--max-time-s", type=float, default=120.0)
    smoke.add_argument("--max-per-class", type=int, default=64)
    smoke.add_argument("--seed", type=int, default=13)
    smoke.add_argument("--bundled-fixture", action="store_true")
    smoke.set_defaults(func=_command_smoke)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
