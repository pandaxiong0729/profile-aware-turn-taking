"""Dependency-free annotation server for SBCSAE event review."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse

from .utils import read_jsonl


LABELS = {"C", "BC", "T", "I", "NA"}
ISSUE_STATUSES = {"UNCERTAIN", "BAD_TARGET", "AUDIO_ISSUE"}
MAX_REQUEST_BYTES = 10 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AnnotationStore:
    """Validate and persist per-reviewer annotations with atomic writes."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        manifest = self.dataset_dir / "annotation_manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing annotation manifest: {manifest}")
        self.events_by_conversation: dict[str, set[str]] = {}
        for row in read_jsonl(manifest):
            conversation_id = str(row["conversation_id"])
            self.events_by_conversation.setdefault(conversation_id, set()).add(
                str(row["event_id"])
            )
        self.review_dir = self.dataset_dir / "human_reviews"
        self.audit_path = self.review_dir / "audit_log.jsonl"
        self.lock = threading.RLock()

    def conversations(self) -> list[dict[str, Any]]:
        return [
            {"conversation_id": conversation_id, "events": len(event_ids)}
            for conversation_id, event_ids in sorted(self.events_by_conversation.items())
        ]

    def _validate_identity(self, reviewer: str, conversation_id: str) -> tuple[str, str]:
        clean_reviewer = reviewer.strip()
        if not clean_reviewer or len(clean_reviewer) > 100:
            raise ValueError("reviewer is required and must be at most 100 characters")
        if conversation_id not in self.events_by_conversation:
            raise ValueError(f"unknown conversation_id: {conversation_id}")
        return clean_reviewer, conversation_id

    def _review_path(self, reviewer: str, conversation_id: str) -> Path:
        reviewer, conversation_id = self._validate_identity(reviewer, conversation_id)
        readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", reviewer).strip("._-")[:32]
        readable = readable or "reviewer"
        digest = hashlib.sha256(reviewer.encode("utf-8")).hexdigest()[:16]
        return self.review_dir / conversation_id / f"{readable}_{digest}.json"

    def load(self, reviewer: str, conversation_id: str) -> dict[str, Any]:
        reviewer, conversation_id = self._validate_identity(reviewer, conversation_id)
        path = self._review_path(reviewer, conversation_id)
        with self.lock:
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                payload = {
                    "schema_version": "1.0",
                    "conversation_id": conversation_id,
                    "reviewer": reviewer,
                    "updated_at": None,
                    "last_position": 0,
                    "reviews": {},
                }
        return {
            **payload,
            "reviews": [
                {"event_id": event_id, **value}
                for event_id, value in sorted(payload.get("reviews", {}).items())
            ],
        }

    def _normalise_review(
        self,
        row: dict[str, Any],
        *,
        reviewer: str,
        conversation_id: str,
    ) -> tuple[str, dict[str, Any]]:
        event_id = str(row.get("event_id") or row.get("sample_id") or "")
        if event_id not in self.events_by_conversation[conversation_id]:
            raise ValueError(f"event_id does not belong to {conversation_id}: {event_id}")
        label = str(row.get("label") or row.get("human_label") or "").upper()
        status = str(row.get("status") or ("OK" if label else "")).upper()
        if label:
            if label not in LABELS:
                raise ValueError(f"invalid label for {event_id}: {label}")
            status = "OK"
        elif status not in ISSUE_STATUSES:
            raise ValueError(f"review for {event_id} needs a label or valid issue status")
        return event_id, {
            "label": label or None,
            "status": status,
            "notes": str(row.get("notes") or row.get("note") or "")[:4000],
            "reviewer": reviewer,
            "updated_at": str(row.get("updated_at") or _now()),
        }

    def save(
        self,
        *,
        reviewer: str,
        conversation_id: str,
        reviews: Iterable[dict[str, Any]],
        last_position: int | None = None,
    ) -> dict[str, Any]:
        reviewer, conversation_id = self._validate_identity(reviewer, conversation_id)
        normalised = [
            self._normalise_review(
                row, reviewer=reviewer, conversation_id=conversation_id
            )
            for row in reviews
        ]
        path = self._review_path(reviewer, conversation_id)
        with self.lock:
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                payload = {
                    "schema_version": "1.0",
                    "conversation_id": conversation_id,
                    "reviewer": reviewer,
                    "last_position": 0,
                    "reviews": {},
                }
            audit_rows: list[dict[str, Any]] = []
            for event_id, value in normalised:
                previous = payload["reviews"].get(event_id)
                payload["reviews"][event_id] = value
                audit_rows.append(
                    {
                        "saved_at": _now(),
                        "conversation_id": conversation_id,
                        "reviewer": reviewer,
                        "event_id": event_id,
                        "previous": previous,
                        "current": value,
                    }
                )
            if last_position is not None:
                payload["last_position"] = max(
                    0,
                    min(
                        int(last_position),
                        len(self.events_by_conversation[conversation_id]) - 1,
                    ),
                )
            payload["updated_at"] = _now()
            _atomic_write_json(path, payload)
            if audit_rows:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as handle:
                    for audit_row in audit_rows:
                        handle.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return {
            "saved": len(normalised),
            "total_reviews": len(payload["reviews"]),
            "last_position": payload["last_position"],
            "updated_at": payload["updated_at"],
        }

    def export(self, reviewer: str, conversation_id: str) -> dict[str, Any]:
        return self.load(reviewer, conversation_id)


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    store: AnnotationStore
    server_version = "SBCSAEAnnotation/1.0"

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: HTTPStatus) -> None:
        self._json({"error": message}, status)

    def _query_identity(self, query: dict[str, list[str]]) -> tuple[str, str]:
        reviewer = query.get("reviewer", [""])[0]
        conversation_id = query.get("conversation_id", [""])[0]
        return reviewer, conversation_id

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        length = int(raw_length)
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("invalid request size")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self._json({"ok": True, "dataset": self.store.dataset_dir.name})
                return
            if parsed.path == "/api/conversations":
                self._json({"conversations": self.store.conversations()})
                return
            if parsed.path == "/api/reviews":
                reviewer, conversation_id = self._query_identity(query)
                self._json(self.store.load(reviewer, conversation_id))
                return
            if parsed.path == "/api/export":
                reviewer, conversation_id = self._query_identity(query)
                payload = self.store.export(reviewer, conversation_id)
                body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                filename = f"{conversation_id}_reviews.json"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{filename}"'
                )
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self._serve_static(parsed.path)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
            self._error(str(error), HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/reviews":
            self._error("not found", HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
            reviews = payload.get("reviews", [])
            if not isinstance(reviews, list):
                raise ValueError("reviews must be a list")
            result = self.store.save(
                reviewer=str(payload.get("reviewer", "")),
                conversation_id=str(payload.get("conversation_id", "")),
                reviews=reviews,
                last_position=payload.get("last_position"),
            )
            self._json(result)
        except (ValueError, json.JSONDecodeError) as error:
            self._error(str(error), HTTPStatus.BAD_REQUEST)

    def _serve_static(self, request_path: str) -> None:
        relative = "review.html" if request_path in {"", "/"} else unquote(
            request_path.lstrip("/")
        )
        allowed = (
            relative == "review.html"
            or relative == "README.md"
            or relative.startswith("review_data/")
            or relative.startswith("audio_clips/")
        )
        if not allowed:
            self._error("not found", HTTPStatus.NOT_FOUND)
            return
        target = (self.store.dataset_dir / relative).resolve()
        try:
            target.relative_to(self.store.dataset_dir)
        except ValueError:
            self._error("invalid path", HTTPStatus.BAD_REQUEST)
            return
        if not target.is_file():
            self._error("not found", HTTPStatus.NOT_FOUND)
            return
        size = target.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            match = re.fullmatch(r"(\d*)-(\d*)", range_header[6:])
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), size - 1)
                if 0 <= start <= end < size:
                    status = HTTPStatus.PARTIAL_CONTENT
        content_length = end - start + 1
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".js":
            content_type = "application/javascript"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header(
            "Cache-Control",
            "public, max-age=3600" if relative.startswith("audio_clips/") else "no-store",
        )
        self.end_headers()
        with target.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def make_server(
    dataset_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    store = AnnotationStore(dataset_dir)
    handler = type(
        "BoundAnnotationRequestHandler",
        (AnnotationRequestHandler,),
        {"store": store},
    )
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the SBCSAE annotation UI")
    parser.add_argument(
        "--dataset-dir", default="data/processed/sbcsae_turn_events_v1"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = make_server(args.dataset_dir, host=args.host, port=args.port)
    print(f"Annotation server: http://{args.host}:{server.server_port}/", flush=True)
    print(f"Dataset: {Path(args.dataset_dir).resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["AnnotationStore", "make_server"]
