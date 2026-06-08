"""Build the composite benchmark artwork required by the manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "paper" / "images"
PANELS = (
    ("circuit-2.png", "(a) Circuit"),
    ("formula-5.png", "(b) Formula"),
    ("french-4.png", "(c) Text"),
    ("graph1-5.png", "(d) Graph"),
    ("handwr2-8.png", "(e) Handwriting"),
    ("large-8.png", "(f) Large glyph"),
    ("symbol-6.png", "(g) Symbols"),
    ("table1-3.png", "(h) Table"),
)


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.35))
    for axis, (filename, label) in zip(axes.flat, PANELS, strict=True):
        with Image.open(IMAGE_DIR / filename) as image:
            axis.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
        axis.set_title(label, fontsize=9, pad=4)
        axis.set_axis_off()
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.96,
        wspace=0.06,
        hspace=0.18,
    )
    fig.savefig(IMAGE_DIR / "Figure_4.pdf", bbox_inches="tight")
    fig.savefig(IMAGE_DIR / "Figure_4.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
