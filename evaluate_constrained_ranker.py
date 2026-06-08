"""Compare CNN and index ordering for the constrained Hamming-1 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from abm_rdh import build_mapping_tables, capacity_bits, evaluate
from ml_candidate_ranker import CandidateRanker
from run_experiments import load_binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ml_ranker_results/validation_ranker.pt"),
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=["table1-3.png", "french-4.png"],
    )
    parser.add_argument("--payload-fraction", type=float, default=0.75)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml_ranker_results/constrained_comparison.json"),
    )
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    ranker = CandidateRanker.load(args.model)
    rng = np.random.default_rng(args.seed)
    rows = []
    for filename in args.images:
        image = load_binary(args.image_dir / filename)
        baseline_tables = build_mapping_tables(image, policy="hamming1")
        cnn_tables = build_mapping_tables(
            image,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        )
        maximum = min(
            capacity_bits(image, baseline_tables),
            capacity_bits(image, cnn_tables),
        )
        payload = int(maximum * args.payload_fraction)
        message = rng.integers(0, 2, size=payload, dtype=np.uint8).tolist()
        _, _, _, baseline = evaluate(image, message, policy="hamming1")
        _, _, _, cnn = evaluate(
            image,
            message,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        )
        rows.append(
            {
                "image": Path(filename).stem,
                "payload_bits": payload,
                "baseline_drd": baseline["drd"],
                "cnn_drd": cnn["drd"],
                "drd_reduction_percent": 100.0
                * (float(baseline["drd"]) - float(cnn["drd"]))
                / float(baseline["drd"]),
                "baseline_psnr_db": baseline["psnr_db"],
                "cnn_psnr_db": cnn["psnr_db"],
                "reversible": cnn["reversible"],
                "message_exact": cnn["message_exact"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
