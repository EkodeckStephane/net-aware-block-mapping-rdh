"""Train and evaluate the Random-Forest uniform-block agent."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ml_uniform_agent import find_safe_uniform_blocks, train_uniform_agent
from run_experiments import DEFAULT_IMAGES, load_binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml_uniform_results"))
    parser.add_argument("--probability-threshold", type=float, default=0.7)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    named_images = [
        (Path(filename).stem, load_binary(args.image_dir / filename))
        for filename in DEFAULT_IMAGES
    ]
    agent, metrics = train_uniform_agent(named_images)
    agent.save(args.output_dir / "uniform_agent.joblib")

    capacities = []
    for name, image in named_images:
        candidates = find_safe_uniform_blocks(
            image,
            agent,
            probability_threshold=args.probability_threshold,
        )
        capacities.append({"image": name, "candidate_blocks": len(candidates)})

    report = {
        "cross_validated_metrics": asdict(metrics),
        "probability_threshold": args.probability_threshold,
        "capacities": capacities,
        "mean_candidate_blocks": sum(row["candidate_blocks"] for row in capacities)
        / len(capacities),
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
