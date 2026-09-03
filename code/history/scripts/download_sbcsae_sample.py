"""Download a user-authorized SBCSAE WAV URL and validate that it is PCM WAV.

Official media hosts may require a browser login. This helper deliberately accepts a
URL instead of embedding credentials or scraping an authenticated session.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="direct authorized WAV download URL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temporary:
        temporary_path = Path(temporary.name)
    try:
        request = urllib.request.Request(args.url, headers={"User-Agent": "profile-turntaking/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        with wave.open(str(temporary_path), "rb") as wav:
            if wav.getnframes() <= 0 or wav.getsampwidth() != 2:
                raise ValueError("Downloaded file is not a non-empty 16-bit PCM WAV")
            print(
                f"validated WAV: channels={wav.getnchannels()} rate={wav.getframerate()} "
                f"seconds={wav.getnframes() / wav.getframerate():.2f}"
            )
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
