"""Download a random set of sample photos (Unsplash photos via Lorem Picsum) into images/samples.

Usage:  python download_samples.py [--count 300] [--width 640]
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

LIST_URL = "https://picsum.photos/v2/list"
OUT_DIR = Path(__file__).resolve().parent / "images" / "samples"


def fetch_catalog() -> list[dict]:
    photos, page = [], 1
    while True:
        r = requests.get(LIST_URL, params={"page": page, "limit": 100}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            return photos
        photos += batch
        page += 1


def download(photo: dict, width: int) -> str:
    path = OUT_DIR / f"picsum_{photo['id']}.jpg"
    if path.exists():
        return "skipped"
    height = round(width * photo["height"] / photo["width"])
    try:
        r = requests.get(f"https://picsum.photos/id/{photo['id']}/{width}/{height}", timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  failed {photo['id']}: {e}")
        return "failed"
    path.write_bytes(r.content)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = fetch_catalog()
    picks = random.Random(args.seed).sample(catalog, min(args.count, len(catalog)))
    print(f"Catalog has {len(catalog)} photos; fetching {len(picks)} into {OUT_DIR}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: download(p, args.width), picks))
    print({s: results.count(s) for s in ("downloaded", "skipped", "failed")})


if __name__ == "__main__":
    main()
