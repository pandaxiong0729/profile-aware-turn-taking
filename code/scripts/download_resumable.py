"""Download a large HTTP file with resumable parallel byte ranges.

The downloader is intended for public research corpora hosted on servers that
advertise ``Accept-Ranges: bytes``. Each range is stored separately so a network
failure or process restart only resumes the incomplete pieces.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


USER_AGENT = "profile-turntaking-data-prep/0.1"


@dataclass(frozen=True)
class ByteRange:
    index: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def _request(url: str, *, method: str = "GET", byte_range: str | None = None):
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = byte_range
    return urllib.request.Request(url, headers=headers, method=method)


def probe_size(url: str, timeout: int) -> tuple[int, str | None]:
    with urllib.request.urlopen(_request(url, method="HEAD"), timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is None:
            raise RuntimeError("Server did not provide Content-Length")
        if "bytes" not in response.headers.get("Accept-Ranges", "").lower():
            raise RuntimeError("Server does not advertise byte-range support")
        return int(length), response.headers.get("ETag")


def split_ranges(total_bytes: int, workers: int) -> list[ByteRange]:
    part_size = (total_bytes + workers - 1) // workers
    ranges: list[ByteRange] = []
    for index in range(workers):
        start = index * part_size
        if start >= total_bytes:
            break
        ranges.append(ByteRange(index, start, min(total_bytes - 1, start + part_size - 1)))
    return ranges


def part_path(parts_dir: Path, item: ByteRange) -> Path:
    return parts_dir / f"part-{item.index:04d}"


def download_range(
    url: str,
    item: ByteRange,
    destination: Path,
    *,
    timeout: int,
    retries: int,
) -> None:
    current_size = destination.stat().st_size if destination.exists() else 0
    if current_size > item.size:
        raise RuntimeError(f"Oversized partial file: {destination}")
    consecutive_failures = 0
    while current_size < item.size:
        absolute_start = item.start + current_size
        try:
            request = _request(url, byte_range=f"bytes={absolute_start}-{item.end}")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"Expected HTTP 206 for range {item.index}, got {response.status}"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {absolute_start}-"):
                    raise RuntimeError(f"Unexpected Content-Range: {content_range!r}")
                before = current_size
                with destination.open("ab") as handle:
                    while current_size < item.size:
                        block = response.read(min(4 * 1024 * 1024, item.size - current_size))
                        if not block:
                            break
                        handle.write(block)
                        current_size += len(block)
                if current_size == before:
                    raise RuntimeError(f"Range {item.index} returned no bytes")
                consecutive_failures = 0
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            consecutive_failures += 1
            if consecutive_failures > retries:
                raise RuntimeError(
                    f"Range {item.index} failed {consecutive_failures} times"
                ) from exc
            time.sleep(min(30, 2**min(consecutive_failures, 5)))


def assemble(output: Path, parts_dir: Path, ranges: list[ByteRange], total_bytes: int) -> None:
    temporary = output.with_name(f"{output.name}.assembling")
    with temporary.open("wb") as destination:
        for item in ranges:
            source_path = part_path(parts_dir, item)
            if source_path.stat().st_size != item.size:
                raise RuntimeError(f"Incomplete part during assembly: {source_path}")
            with source_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
    if temporary.stat().st_size != total_bytes:
        raise RuntimeError("Assembled file size does not match the server Content-Length")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--expected-bytes", type=int)
    parser.add_argument("--progress-seconds", type=int, default=10)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("workers must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_bytes, etag = probe_size(args.url, args.timeout)
    if args.expected_bytes is not None and total_bytes != args.expected_bytes:
        raise RuntimeError(
            f"Expected {args.expected_bytes} bytes, but server reports {total_bytes}"
        )
    if output.exists() and output.stat().st_size == total_bytes:
        print(json.dumps({"status": "already_complete", "bytes": total_bytes, "etag": etag}))
        return

    ranges = split_ranges(total_bytes, args.workers)
    parts_dir = output.with_name(f"{output.name}.parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    first_part = part_path(parts_dir, ranges[0])
    if output.exists():
        prefix_size = output.stat().st_size
        if first_part.exists():
            raise RuntimeError("Both a partial output and first range part exist; choose one")
        if prefix_size > ranges[0].size:
            raise RuntimeError("Existing partial output is larger than the first byte range")
        output.replace(first_part)

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [
            executor.submit(
                download_range,
                args.url,
                item,
                part_path(parts_dir, item),
                timeout=args.timeout,
                retries=args.retries,
            )
            for item in ranges
        ]
        while not all(future.done() for future in futures):
            downloaded = sum(
                min(item.size, part_path(parts_dir, item).stat().st_size)
                if part_path(parts_dir, item).exists()
                else 0
                for item in ranges
            )
            elapsed = max(1e-6, time.monotonic() - started)
            print(
                json.dumps(
                    {
                        "status": "downloading",
                        "bytes": downloaded,
                        "total_bytes": total_bytes,
                        "percent": round(100 * downloaded / total_bytes, 2),
                        "mib_per_second": round(downloaded / elapsed / (1024 * 1024), 2),
                    }
                ),
                flush=True,
            )
            time.sleep(args.progress_seconds)
        for future in futures:
            future.result()

    assemble(output, parts_dir, ranges, total_bytes)
    shutil.rmtree(parts_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "etag": etag,
            }
        )
    )


if __name__ == "__main__":
    main()
