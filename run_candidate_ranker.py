"""Train and evaluate the CNN candidate-cost ranker."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ml_candidate_ranker import train_candidate_ranker
from run_experiments import DEFAULT_IMAGES, load_binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml_ranker_results"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    named_images = [
        (Path(filename).stem, load_binary(args.image_dir / filename))
        for filename in DEFAULT_IMAGES
    ]
    ranker, validation_ranker, metrics = train_candidate_ranker(
        named_images,
        epochs=args.epochs,
    )
    ranker.save(args.output_dir / "candidate_ranker.pt")
    validation_ranker.save(args.output_dir / "validation_ranker.pt")
    report = asdict(metrics)
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
