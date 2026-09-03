from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from profile_turntaking.annotation_server import AnnotationStore, make_server


def _dataset(tmp_path: Path) -> Path:
    rows = [
        {"event_id": "SBC005-event-1", "conversation_id": "SBC005"},
        {"event_id": "SBC005-event-2", "conversation_id": "SBC005"},
    ]
    (tmp_path / "annotation_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (tmp_path / "review.html").write_text("<h1>review</h1>", encoding="utf-8")
    (tmp_path / "audio_clips").mkdir()
    (tmp_path / "audio_clips" / "sample.wav").write_bytes(b"RIFF-test-audio")
    return tmp_path


def test_store_writes_real_file_and_restores_it(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    store = AnnotationStore(dataset)
    saved = store.save(
        reviewer="R01",
        conversation_id="SBC005",
        reviews=[{"event_id": "SBC005-event-1", "label": "T", "notes": "ok"}],
        last_position=1,
    )

    assert saved["saved"] == 1
    files = list((dataset / "human_reviews" / "SBC005").glob("*.json"))
    assert len(files) == 1
    restored = AnnotationStore(dataset).load("R01", "SBC005")
    assert restored["last_position"] == 1
    assert restored["reviews"] == [
        {
            "event_id": "SBC005-event-1",
            "label": "T",
            "status": "OK",
            "notes": "ok",
            "reviewer": "R01",
            "updated_at": restored["reviews"][0]["updated_at"],
        }
    ]
    assert (dataset / "human_reviews" / "audit_log.jsonl").is_file()


def test_store_rejects_unknown_event(tmp_path: Path) -> None:
    store = AnnotationStore(_dataset(tmp_path))
    with pytest.raises(ValueError, match="does not belong"):
        store.save(
            reviewer="R01",
            conversation_id="SBC005",
            reviews=[{"event_id": "wrong", "label": "C"}],
        )


def test_http_save_reload_and_audio_range(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    server = make_server(dataset, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        payload = json.dumps(
            {
                "reviewer": "R02",
                "conversation_id": "SBC005",
                "last_position": 1,
                "reviews": [{"event_id": "SBC005-event-2", "label": "I"}],
            }
        ).encode()
        request = Request(
            base + "/api/reviews",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            result = json.load(response)
        assert result["total_reviews"] == 1

        with urlopen(
            base + "/api/reviews?conversation_id=SBC005&reviewer=R02"
        ) as response:
            restored = json.load(response)
        assert restored["last_position"] == 1
        assert restored["reviews"][0]["event_id"] == "SBC005-event-2"
        assert restored["reviews"][0]["label"] == "I"

        with urlopen(
            base + "/api/export?conversation_id=SBC005&reviewer=R02"
        ) as response:
            exported = json.load(response)
            assert response.headers["Content-Disposition"] == (
                'attachment; filename="SBC005_reviews.json"'
            )
        assert exported["reviewer"] == "R02"
        assert exported["reviews"][0]["label"] == "I"

        audio_request = Request(
            base + "/audio_clips/sample.wav", headers={"Range": "bytes=0-3"}
        )
        with urlopen(audio_request) as response:
            assert response.status == 206
            assert response.read() == b"RIFF"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
