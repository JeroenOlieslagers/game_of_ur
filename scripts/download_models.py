#!/usr/bin/env python3
"""Download selected public RoyalUr solved maps from Hugging Face."""

from __future__ import annotations

import argparse
import pathlib
import time
import urllib.request


REPOSITORY = "https://huggingface.co/sothatsit/RoyalUrModels/resolve/main"
FILES = {
    "blitz": "blitz.rgu",
    "masters": "masters3d.rgu",
    "finkel": "finkel.rgu",
    # Lamont's published f64 Finkel map, used as an external reference to check
    # our own Finkel solve against.
    "finkel_f64": "finkel_f64.rgu",
}


def download(filename: str, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "RoyalUr-paper-revision/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    url = f"{REPOSITORY}/{filename}?download=true"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = existing > 0 and response.status == 206
        if existing and not resumed:
            existing = 0
        content_length = int(response.headers.get("Content-Length", "0"))
        total = existing + content_length if content_length else 0
        mode = "ab" if resumed else "wb"
        downloaded = existing
        started = time.monotonic()
        last_report = started
        with partial.open(mode) as output:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    elapsed = max(now - started, 1e-9)
                    rate = (downloaded - existing) / elapsed / 1_000_000
                    total_text = f"/{total / 1_000_000:.1f} MB" if total else ""
                    print(
                        f"{filename}: {downloaded / 1_000_000:.1f} MB"
                        f"{total_text} at {rate:.1f} MB/s",
                        flush=True,
                    )
                    last_report = now

    if total and downloaded != total:
        raise RuntimeError(
            f"Incomplete download for {filename}: got {downloaded}, expected {total}"
        )
    partial.replace(destination)
    print(f"saved {destination} ({downloaded} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=sorted(FILES))
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "models",
    )
    args = parser.parse_args()
    for model in args.models:
        filename = FILES[model]
        download(filename, args.output_dir / filename)


if __name__ == "__main__":
    main()
