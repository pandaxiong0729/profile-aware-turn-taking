from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path


CONVERSATION_ID = "SBC005"
PACKAGE_NAME = "sbcsae_turn_events_SBC005"
JSONL_FILES = (
    "annotation_manifest.jsonl",
    "event_candidates.jsonl",
    "interaction_structures.jsonl",
    "ipus.jsonl",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def filter_jsonl(source: Path, destination: Path) -> list[dict]:
    rows: list[dict] = []
    with source.open("r", encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for line in reader:
            row = json.loads(line)
            if row.get("conversation_id") == CONVERSATION_ID:
                rows.append(row)
                writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_package(source_dir: Path, output_zip: Path) -> dict:
    source_dir = source_dir.resolve()
    output_zip = output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sbc005_teacher_package_") as temp_name:
        package_dir = Path(temp_name) / PACKAGE_NAME
        package_dir.mkdir()
        (package_dir / "audio_clips").mkdir()
        (package_dir / "review_data").mkdir()
        (package_dir / "human_reviews").mkdir()

        filtered: dict[str, list[dict]] = {}
        for filename in JSONL_FILES:
            filtered[filename] = filter_jsonl(
                source_dir / filename, package_dir / filename
            )

        manifest = filtered["annotation_manifest.jsonl"]
        event_ids = {str(row["event_id"]) for row in manifest}
        if len(manifest) != 894 or len(event_ids) != 894:
            raise RuntimeError(
                f"Expected 894 unique SBC005 events, got {len(manifest)} rows and {len(event_ids)} ids"
            )

        copied_audio: list[Path] = []
        for row in manifest:
            relative = Path(str(row["audio_path"]))
            if relative.is_absolute() or relative.parts[:1] != ("audio_clips",):
                raise RuntimeError(f"Invalid relative audio path: {relative}")
            source_audio = (source_dir / relative).resolve()
            source_audio.relative_to(source_dir)
            if not source_audio.is_file():
                raise FileNotFoundError(source_audio)
            destination_audio = package_dir / relative
            shutil.copy2(source_audio, destination_audio)
            copied_audio.append(destination_audio)

        shutil.copy2(source_dir / "review.html", package_dir / "review.html")
        shutil.copy2(
            source_dir / "review_data" / f"{CONVERSATION_ID}.js",
            package_dir / "review_data" / f"{CONVERSATION_ID}.js",
        )
        (package_dir / "review_data" / "index.js").write_text(
            'window.REVIEW_CONVERSATIONS=[{"conversation_id":"SBC005","events":894}];\n',
            encoding="utf-8",
        )

        event_rows = filtered["event_candidates.jsonl"]
        structure_rows = filtered["interaction_structures.jsonl"]
        ipu_rows = filtered["ipus.jsonl"]
        source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
        conversation = next(
            row
            for row in source_summary["per_conversation"]
            if row["conversation_id"] == CONVERSATION_ID
        )
        candidate_counts = Counter(str(row["candidate_label"]) for row in event_rows)
        structure_counts = Counter(str(row["structure"]) for row in structure_rows)
        confidence_counts = Counter(
            str(row.get("candidate_confidence", "unknown")) for row in event_rows
        )
        summary = {
            "schema_version": source_summary.get("schema_version", "1.0"),
            "package": "SBC005_teacher_subset",
            "conversation_id": CONVERSATION_ID,
            "source_dataset_conversations": source_summary["conversations"],
            "source_dataset_events": source_summary["events"],
            "conversations": 1,
            "duration_s": conversation["duration_s"],
            "duration_minutes": conversation["duration_s"] / 60,
            "ipus": len(ipu_rows),
            "structures": len(structure_rows),
            "events": len(event_rows),
            "review_audio_files": len(copied_audio),
            "candidate_label_counts": dict(candidate_counts),
            "structure_counts": dict(structure_counts),
            "confidence_counts": dict(confidence_counts),
            "per_conversation": [conversation],
        }
        write_json(package_dir / "summary.json", summary)
        write_json(
            package_dir / "review_site.json",
            {
                "review_page": "review.html",
                "conversations": 1,
                "events": len(event_rows),
                "portable": True,
            },
        )

        all_relative = all(not Path(str(row["audio_path"])).is_absolute() for row in manifest)
        all_audio_present = len(copied_audio) == len(manifest) and all(
            path.is_file() for path in copied_audio
        )
        verification = {
            "verified": all(
                (
                    len(manifest) == 894,
                    len(event_ids) == 894,
                    set(candidate_counts) == {"C", "BC", "T", "I", "NA"},
                    all_relative,
                    all_audio_present,
                    (package_dir / "review.html").is_file(),
                )
            ),
            "checks": {
                "only_SBC005": all(
                    row.get("conversation_id") == CONVERSATION_ID
                    for rows in filtered.values()
                    for row in rows
                ),
                "894_unique_events": len(manifest) == len(event_ids) == 894,
                "all_five_labels_present": set(candidate_counts)
                == {"C", "BC", "T", "I", "NA"},
                "relative_audio_paths": all_relative,
                "one_audio_per_event": all_audio_present,
                "review_page_present": (package_dir / "review.html").is_file(),
            },
            "events": len(manifest),
            "audio_files": len(copied_audio),
            "candidate_label_counts": dict(candidate_counts),
        }
        if not verification["verified"] or not all(verification["checks"].values()):
            raise RuntimeError(f"Package verification failed: {verification}")
        write_json(package_dir / "verification.json", verification)

        package_mib = sum(path.stat().st_size for path in package_dir.rglob("*") if path.is_file()) / 1024**2
        readme = (source_dir / "README.md").read_text(encoding="utf-8")
        readme = readme.replace(
            "# SBCSAE 五分类标注数据",
            "# SBCSAE 五分类标注数据（SBC005 教师查看版）\n\n"
            "这个压缩包只包含SBC005会话，所有JSONL、短音频和标注页面相互对应，可以直接检查。\n\n"
            "最简单的查看方法：解压后直接打开 `review.html`。页面可以播放音频、显示事件位置并选择五分类标签；需要保存时点击“导出结果”。",
            1,
        )
        replacements = {
            "| 双人会话 | 16段 |": "| 双人会话 | 1段（SBC005） |",
            "| 会话总时长 | 约6.43小时 |": "| 会话总时长 | 约20.47分钟 |",
            "| 需要标注的位置 | 16,234个 |": "| 需要标注的位置 | 894个 |",
            "| 对应的短音频 | 16,234个 |": "| 对应的短音频 | 894个 |",
            "| 数据大小 | 约4.92 GiB |": f"| 解压后数据大小 | 约{package_mib:.1f} MiB |",
            "sbcsae_turn_events_v1/": f"{PACKAGE_NAME}/",
        }
        for old, new in replacements.items():
            readme = readme.replace(old, new)
        (package_dir / "README.md").write_text(readme, encoding="utf-8")

        output_zip.unlink(missing_ok=True)
        with zipfile.ZipFile(
            output_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            archive.writestr(f"{PACKAGE_NAME}/human_reviews/", "")
            for path in sorted(package_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package_dir.parent).as_posix())

    with zipfile.ZipFile(output_zip) as archive:
        names = archive.namelist()
        audio_names = [name for name in names if name.endswith(".wav")]
        bad_audio = [name for name in audio_names if "/SBC005-event-" not in name]
        bad_crc = archive.testzip()
    required = {
        f"{PACKAGE_NAME}/README.md",
        f"{PACKAGE_NAME}/review.html",
        f"{PACKAGE_NAME}/annotation_manifest.jsonl",
        f"{PACKAGE_NAME}/event_candidates.jsonl",
        f"{PACKAGE_NAME}/interaction_structures.jsonl",
        f"{PACKAGE_NAME}/ipus.jsonl",
        f"{PACKAGE_NAME}/summary.json",
        f"{PACKAGE_NAME}/verification.json",
        f"{PACKAGE_NAME}/review_site.json",
        f"{PACKAGE_NAME}/review_data/index.js",
        f"{PACKAGE_NAME}/review_data/SBC005.js",
        f"{PACKAGE_NAME}/human_reviews/",
    }
    missing_required = sorted(required.difference(names))
    if len(audio_names) != 894 or bad_audio or bad_crc or missing_required:
        raise RuntimeError(
            "Archive check failed: "
            f"audio={len(audio_names)}, bad_audio={bad_audio[:3]}, "
            f"bad_crc={bad_crc}, missing={missing_required}"
        )
    if output_zip.stat().st_size >= 1_000_000_000:
        raise RuntimeError(f"Archive exceeds 1 GB: {output_zip.stat().st_size} bytes")
    return {
        "zip_path": str(output_zip),
        "zip_bytes": output_zip.stat().st_size,
        "zip_mib": output_zip.stat().st_size / 1024**2,
        "zip_sha256": sha256(output_zip),
        "archive_files": len(names),
        "audio_files": len(audio_names),
        "events": 894,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SBC005 teacher data package")
    parser.add_argument(
        "--source-dir", default="data/processed/sbcsae_turn_events_v1"
    )
    parser.add_argument(
        "--output-zip", default="artifacts/share/SBC005_teacher_package.zip"
    )
    args = parser.parse_args()
    result = build_package(Path(args.source_dir), Path(args.output_zip))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
