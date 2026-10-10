"""Small-detail test: can the model tell apart drawing sheets that differ only in their text?

Generates identical floor-plan sheets whose only difference is the drawing title (in small or
large text), then searches for each title at different image token budgets and reports how often
the correct sheet ranks first.

Usage:  python test_small_details.py [--budgets 280 1120]
Sheets are written to images/test_drawings/ so you can also try them in the app (press Re-index).
"""
from __future__ import annotations

import argparse
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer

from engine import DEFAULT_IMAGE_DIR, MODEL_ID, QUERY_PREFIX, pick_device

OUT_DIR = DEFAULT_IMAGE_DIR / "test_drawings"
W, H = 2400, 1700
# Title font size in pixels on a 2400 px wide sheet. "large" is a banner across the top;
# the rest sit in the title block like on a real drawing.
TEXT_SIZES = {"large": 96, "small": 26, "tiny": 16, "micro": 11, "nano": 8}

# (title on the sheet, how a user might search for it)
SHEETS = [
    ("ELECTRICAL POWER LAYOUT - GROUND FLOOR", "electrical power layout"),
    ("PLUMBING AND DRAINAGE LAYOUT", "plumbing and drainage drawing"),
    ("FIRE ALARM SYSTEM LAYOUT", "fire alarm system plan"),
    ("HVAC DUCTWORK LAYOUT", "HVAC ductwork drawing"),
    ("STRUCTURAL FOUNDATION PLAN", "structural foundation plan"),
    ("LANDSCAPE IRRIGATION PLAN", "landscape irrigation drawing"),
    ("LIGHTING LAYOUT - FIRST FLOOR", "lighting layout first floor"),
    ("CCTV AND SECURITY LAYOUT", "CCTV security cameras plan"),
    ("ROOF DRAINAGE PLAN", "roof drainage plan"),
    ("FURNITURE LAYOUT", "furniture layout drawing"),
    ("KITCHEN EQUIPMENT LAYOUT", "kitchen equipment layout"),
    ("BASEMENT CAR PARKING LAYOUT", "basement car parking plan"),
]


def font(size: int, bold: bool = False):
    for name in (("arialbd.ttf", "Arial Bold.ttf") if bold else ("arial.ttf", "Arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def base_plan() -> Image.Image:
    """One shared floor plan (same for every sheet) so only the text differs."""
    rng = random.Random(42)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, W - 30, H - 30], outline="black", width=4)          # sheet border
    x0, y0, x1, y1 = 140, 140, 1900, 1380                                     # building outline
    d.rectangle([x0, y0, x1, y1], outline="black", width=10)
    for x in range(x0, x1 + 1, 220):                                          # grid lines
        d.line([x, 80, x, H - 260], fill=(170, 170, 170), width=1)
    for y in range(y0, y1 + 1, 220):
        d.line([80, y, 1960, y], fill=(170, 170, 170), width=1)
    walls_x = [x0 + 440, x0 + 880, x0 + 1320]
    for x in walls_x:
        d.line([x, y0, x, y1], fill="black", width=6)
    d.line([x0, y0 + 620, x1, y0 + 620], fill="black", width=6)
    for x in walls_x + [x0 + 60]:                                             # door swings
        for y in (y0 + 620,):
            d.arc([x + 20, y - 90, x + 200, y + 90], 270, 360, fill="black", width=3)
    labels = ["LOBBY", "OFFICE 01", "OFFICE 02", "MEETING", "STORE", "WC", "PANTRY", "CORRIDOR"]
    f = font(24)
    for i, label in enumerate(labels):
        cx = x0 + 220 + (i % 4) * 440
        cy = y0 + 310 + (i // 4) * 620
        d.text((cx, cy), label, fill="black", font=f, anchor="mm")
    for _ in range(40):                                                       # generic symbols
        cx, cy = rng.randint(x0 + 40, x1 - 40), rng.randint(y0 + 40, y1 - 40)
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline="black", width=2)
    d.rectangle([W - 470, H - 260, W - 30, H - 30], outline="black", width=3)  # title block
    d.text((W - 450, H - 90), "PROJECT: OFFICE BUILDING  |  SCALE 1:100", fill="black", font=font(18))
    return img


def make_sheet(base: Image.Image, title: str, size: int) -> Image.Image:
    img = base.copy()
    d = ImageDraw.Draw(img)
    if size == TEXT_SIZES["large"]:   # big title across the top of the sheet
        d.text((W // 2, 85), title, fill="black", font=font(size, bold=True), anchor="mm")
    else:                             # only in the title block, like a real drawing
        y = H - 240
        for line in _wrap(title, 26):
            d.text((W - 450, y), line, fill="black", font=font(size, bold=True))
            y += size + 8
    return img


def _wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + [cur]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=int, nargs="+", default=[70, 280, 1120])
    parser.add_argument("--sizes", nargs="+", default=list(TEXT_SIZES), choices=list(TEXT_SIZES))
    args = parser.parse_args()

    base = base_plan()
    sets = {}
    for name in args.sizes:
        size = TEXT_SIZES[name]
        label = f"{name} {size}px"
        folder = OUT_DIR / f"{name}_text"
        folder.mkdir(parents=True, exist_ok=True)
        imgs = []
        for i, (title, _) in enumerate(SHEETS):
            img = make_sheet(base, title, size)
            img.save(folder / f"sheet_{i + 1:02d}.png")
            imgs.append(img)
        sets[label] = imgs
    print(f"Wrote {len(SHEETS)} sheets x {len(sets)} text sizes to {OUT_DIR}")

    device, dtype = pick_device()
    model = SentenceTransformer(MODEL_ID, device=device, model_kwargs={"torch_dtype": dtype})
    queries = model.encode([QUERY_PREFIX + q for _, q in SHEETS], normalize_embeddings=True)
    queries = np.asarray(queries, dtype=np.float32)

    print(f"\n{len(SHEETS)} sheets per set, identical except for the title. Random guessing gets "
          f"top-1 = {100 / len(SHEETS):.0f}%.\n")
    print(f"{'set':<14}{'tokens':>8}{'top-1':>9}{'top-3':>9}{'avg rank':>10}")
    details = {}
    for label, imgs in sets.items():
        for budget in args.budgets:
            emb = model.encode([{"image": im} for im in imgs], batch_size=4, normalize_embeddings=True,
                               processing_kwargs={"image": {"max_soft_tokens": budget}})
            scores = queries @ np.asarray(emb, dtype=np.float32).T             # [query, sheet]
            ranks = [int((scores[i] > scores[i, i]).sum()) + 1 for i in range(len(SHEETS))]
            details[(label, budget)] = ranks
            top1 = 100 * sum(r == 1 for r in ranks) / len(ranks)
            top3 = 100 * sum(r <= 3 for r in ranks) / len(ranks)
            print(f"{label:<14}{budget:>8}{top1:>8.0f}%{top3:>8.0f}%{np.mean(ranks):>10.1f}")

    if len(details) > 6:
        return   # per-search table is only readable for a few configurations
    print("\nRank of the correct sheet for each search (1 = found first):")
    header = "".join(f"{f'{lbl.split()[0]}/{b}':>12}" for lbl, b in details)
    print(f"{'search':<32}{header}")
    for i, (_, q) in enumerate(SHEETS):
        print(f"{q:<32}" + "".join(f"{details[key][i]:>12}" for key in details))


if __name__ == "__main__":
    main()
