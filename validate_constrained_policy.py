"""Validate the selected distortion-constrained policy over multiple seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from abm_rdh import build_mapping_tables, capacity_bits, evaluate
from run_experiments import DEFAULT_IMAGES, load_binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("constrained_validation")
    )
    parser.add_argument("--payload-fraction", type=float, default=0.75)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for seed_index in range(args.seeds):
        seed = 20260607 + seed_index
        rng = np.random.default_rng(seed)
        for filename in DEFAULT_IMAGES:
            image = load_binary(args.image_dir / filename)
            tables = build_mapping_tables(image, policy="hamming1")
            maximum = capacity_bits(image, tables)
            payload = int(maximum * args.payload_fraction)
            message = rng.integers(0, 2, size=payload, dtype=np.uint8).tolist()
            _, _, _, metrics = evaluate(
                image,
                message,
                policy="hamming1",
            )
            rows.append(
                {
                    "seed": seed,
                    "image": Path(filename).stem,
                    "payload_fraction": args.payload_fraction,
                    "maximum_capacity_bits": maximum,
                    **metrics,
                }
            )

    with (args.output_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    drds = [float(row["drd"]) for row in rows]
    summary = {
        "runs": len(rows),
        "seeds": args.seeds,
        "payload_fraction": args.payload_fraction,
        "mean_capacity_bits": sum(int(row["capacity_bits"]) for row in rows)
        / len(rows),
        "mean_psnr_db": sum(float(row["psnr_db"]) for row in rows) / len(rows),
        "mean_drd": sum(drds) / len(drds),
        "maximum_drd": max(drds),
        "all_drd_below_one": all(value < 1.0 for value in drds),
        "all_reversible": all(bool(row["reversible"]) for row in rows),
        "all_messages_exact": all(bool(row["message_exact"]) for row in rows),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
