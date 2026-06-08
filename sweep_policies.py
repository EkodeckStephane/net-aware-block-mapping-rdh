"""Compare adaptive and distortion-constrained policies reproducibly."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_experiments import run_dataset


CONFIGURATIONS = (
    ("adaptive", 0.05, 1.0),
    ("adaptive", 0.10, 1.0),
    ("adaptive", 0.20, 1.0),
    ("adaptive", 0.50, 1.0),
    ("adaptive", 1.00, 1.0),
    ("hamming1", 0.10, 0.25),
    ("hamming1", 0.10, 0.50),
    ("hamming1", 0.10, 0.75),
    ("hamming1", 0.10, 1.00),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("policy_sweep"))
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for policy, alpha, payload_fraction in CONFIGURATIONS:
        name = f"{policy}__a{alpha:.2f}__p{payload_fraction:.2f}"
        rows = run_dataset(
            args.image_dir,
            args.output_dir / name,
            alpha=alpha,
            seed=args.seed,
            policy=policy,
            payload_fraction=payload_fraction,
        )
        all_rows.extend(rows)

    with (args.output_dir / "all_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    print("policy     alpha payload  mean bits  mean PSNR  mean DRD  DRD max")
    for policy, alpha, payload_fraction in CONFIGURATIONS:
        selected = [
            row
            for row in all_rows
            if row["policy"] == policy
            and row["alpha"] == alpha
            and row["payload_fraction"] == payload_fraction
        ]
        count = len(selected)
        print(
            f"{policy:<10} {alpha:>5.2f} {payload_fraction:>7.2f} "
            f"{sum(row['capacity_bits'] for row in selected) / count:>10.1f} "
            f"{sum(row['psnr_db'] for row in selected) / count:>10.3f} "
            f"{sum(row['drd'] for row in selected) / count:>9.3f} "
            f"{max(row['drd'] for row in selected):>8.3f}"
        )


if __name__ == "__main__":
    main()
