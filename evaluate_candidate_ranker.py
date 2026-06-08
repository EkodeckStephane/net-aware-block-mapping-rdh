"""Compare CNN and Hamming candidate ordering at identical payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from abm_rdh import build_mapping_tables, capacity_bits, evaluate
from ml_candidate_ranker import CandidateRanker
from run_experiments import DEFAULT_IMAGES, load_binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ml_ranker_results/validation_ranker.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("ml_ranker_results/comparison.json"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument(
        "--images",
        nargs="+",
        default=["table1-3.png", "french-4.png"],
    )
    args = parser.parse_args()

    ranker = CandidateRanker.load(args.model)
    rng = np.random.default_rng(args.seed)
    rows = []
    for filename in args.images:
        if filename not in DEFAULT_IMAGES:
            raise ValueError(f"Unknown benchmark image: {filename}")
        image = load_binary(args.image_dir / filename)
        hamming_tables = build_mapping_tables(
            image,
            alpha=args.alpha,
            policy="adaptive",
        )
        cnn_tables = build_mapping_tables(
            image,
            alpha=args.alpha,
            policy="cnn",
            candidate_ranker=ranker,
        )
        hamming_capacity = capacity_bits(image, hamming_tables)
        cnn_capacity = capacity_bits(image, cnn_tables)
        payload = min(hamming_capacity, cnn_capacity)
        message = rng.integers(0, 2, size=payload, dtype=np.uint8).tolist()
        _, _, _, hamming_metrics = evaluate(
            image,
            message,
            alpha=args.alpha,
            policy="adaptive",
        )
        _, _, _, cnn_metrics = evaluate(
            image,
            message,
            alpha=args.alpha,
            policy="cnn",
            candidate_ranker=ranker,
        )
        rows.append(
            {
                "image": Path(filename).stem,
                "payload_bits": payload,
                "hamming_psnr_db": hamming_metrics["psnr_db"],
                "cnn_psnr_db": cnn_metrics["psnr_db"],
                "hamming_drd": hamming_metrics["drd"],
                "cnn_drd": cnn_metrics["drd"],
                "drd_reduction_percent": 100.0
                * (float(hamming_metrics["drd"]) - float(cnn_metrics["drd"]))
                / float(hamming_metrics["drd"]),
                "cnn_reversible": cnn_metrics["reversible"],
                "cnn_message_exact": cnn_metrics["message_exact"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
