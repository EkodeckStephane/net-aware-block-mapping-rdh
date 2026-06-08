"""Run GP-based alpha optimization on the bundled benchmark images."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from ml_alpha_optimizer import best_observation, optimize_alpha
from run_experiments import DEFAULT_IMAGES, load_binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml_alpha_results"))
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    best_rows = []
    for index, filename in enumerate(DEFAULT_IMAGES):
        image = load_binary(args.image_dir / filename)
        observations = optimize_alpha(
            image,
            seed=args.seed + index,
            iterations=args.iterations,
            lambda_weight=args.lambda_weight,
        )
        name = Path(filename).stem
        for observation in observations:
            traces.append({"image": name, **asdict(observation)})
        best_rows.append({"image": name, **asdict(best_observation(observations))})

    with (args.output_dir / "traces.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    with (args.output_dir / "best.json").open("w", encoding="utf-8") as handle:
        json.dump(best_rows, handle, indent=2)

    print("image          alpha  capacity  PSNR (dB)    DRD  objective")
    for row in best_rows:
        print(
            f"{row['image']:<14} {row['alpha']:>5.2f} "
            f"{row['capacity_bits']:>9} {row['psnr_db']:>10.3f} "
            f"{row['drd']:>6.3f} {row['objective']:>10.4f}"
        )


if __name__ == "__main__":
    main()
