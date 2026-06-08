"""Run deterministic experiments on the eight bundled benchmark images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from abm_rdh import as_binary, build_mapping_tables, capacity_bits, evaluate


DEFAULT_IMAGES = (
    "circuit-2.png",
    "formula-5.png",
    "graph1-5.png",
    "handwr2-8.png",
    "large-8.png",
    "symbol-6.png",
    "table1-3.png",
    "french-4.png",
)
DOCUMENT_IMAGE_DIR = Path(__file__).resolve().parent / "paper" / "images"


def load_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return as_binary(np.asarray(image.convert("L")))


def save_binary(path: Path, image: np.ndarray) -> None:
    Image.fromarray(as_binary(image) * 255).save(path)


def run_dataset(
    image_dir: Path,
    output_dir: Path,
    *,
    alpha: float,
    seed: int,
    policy: str = "adaptive",
    payload_fraction: float = 1.0,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for filename in DEFAULT_IMAGES:
        source = image_dir / filename
        image = load_binary(source)
        tables = build_mapping_tables(image, alpha=alpha, policy=policy)
        maximum_capacity = capacity_bits(image, tables)
        payload_bits = int(maximum_capacity * payload_fraction)
        message = rng.integers(0, 2, size=payload_bits, dtype=np.uint8).tolist()
        embedded, restored, _, metrics = evaluate(
            image,
            message,
            alpha=alpha,
            policy=policy,
        )

        name = source.stem
        save_binary(output_dir / f"stego__{name}.png", embedded.stego)
        save_binary(output_dir / f"restored__{name}.png", restored)
        row: dict[str, object] = {
            "image": name,
            "width": image.shape[1],
            "height": image.shape[0],
            "alpha": alpha,
            "policy": policy,
            "payload_fraction": payload_fraction,
            "maximum_capacity_bits": maximum_capacity,
            "seed": seed,
            "mapping_tables": len(embedded.auxiliary.mapping_tables),
            **metrics,
        }
        rows.append(row)

    fieldnames = list(rows[0])
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=DOCUMENT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument(
        "--policy",
        choices=("adaptive", "hamming1"),
        default="adaptive",
    )
    parser.add_argument("--payload-fraction", type=float, default=1.0)
    args = parser.parse_args()

    rows = run_dataset(
        args.image_dir,
        args.output_dir,
        alpha=args.alpha,
        seed=args.seed,
        policy=args.policy,
        payload_fraction=args.payload_fraction,
    )
    print("image          capacity  PSNR (dB)    DRD   reversible  exact")
    for row in rows:
        print(
            f"{row['image']:<14} {row['capacity_bits']:>8}  "
            f"{row['psnr_db']:>9.3f}  {row['drd']:>5.3f}   "
            f"{str(row['reversible']):>10}  {str(row['message_exact']):>5}"
        )


if __name__ == "__main__":
    main()
