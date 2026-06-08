"""Recolor existing dark chart PNG backgrounds to the current navy palette."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [
    ROOT / "outputs" / "landing",
    ROOT / "outputs" / "buy_side_advanced",
]

REPLACEMENTS = {
    (5, 25, 70): (2, 8, 23),
    (6, 20, 47): (6, 22, 65),
    (7, 31, 85): (6, 22, 65),
}


def recolor(path: Path) -> bool:
    image = Image.open(path).convert("RGBA")
    pixels = image.load()
    changed = False
    width, height = image.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            replacement = REPLACEMENTS.get((r, g, b))
            if replacement is not None:
                pixels[x, y] = (*replacement, a)
                changed = True
    if changed:
        image.save(path)
    return changed


def main() -> None:
    changed = 0
    for folder in TARGET_DIRS:
        for path in sorted(folder.glob("*.png")):
            if recolor(path):
                changed += 1
    print(f"Recolored {changed} chart PNGs")


if __name__ == "__main__":
    main()
