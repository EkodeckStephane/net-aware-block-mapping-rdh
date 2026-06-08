"""External net-capacity evaluation on thresholded BOSSbase images."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from abm_rdh import (
    auxiliary_bits,
    build_mapping_tables,
    capacity_bits,
    deserialize_auxiliary,
    drd,
    embed,
    extract,
    optimize_tables_for_net_capacity,
    psnr,
    serialize_auxiliary,
)
from ml_candidate_ranker import CandidateRanker


DEFAULT_BOSSBASE = Path(os.environ.get("BOSSBASE_DIR", "datasets/BOSSbase"))


def load_thresholded(path: Path, threshold: int) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    return (grayscale >= threshold).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_BOSSBASE)
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=Path("ml_ranker_results/candidate_ranker.pt"),
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--payload-fraction", type=float, default=0.75)
    parser.add_argument(
        "--no-net-optimize",
        action="store_true",
        help="Disable per-image mapping-table selection by net capacity.",
    )
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--output", type=Path, default=Path("bossbase_net_results.json")
    )
    args = parser.parse_args()

    paths = sorted(
        args.dataset.glob("*.pgm"),
        key=lambda path: int(path.stem),
    )
    if args.sample_size > len(paths):
        raise ValueError("Requested sample is larger than the dataset")
    rng = np.random.default_rng(args.seed)
    selected = [
        paths[index]
        for index in sorted(
            rng.choice(len(paths), size=args.sample_size, replace=False).tolist()
        )
    ]
    ranker = CandidateRanker.load(args.ranker_model)
    rows = []
    for path in selected:
        image = load_thresholded(path, args.threshold)
        tables = build_mapping_tables(
            image,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        )
        if not args.no_net_optimize:
            tables = optimize_tables_for_net_capacity(
                image,
                tables,
                policy="cnn_hamming1",
            )
        maximum = capacity_bits(image, tables)
        payload_length = int(maximum * args.payload_fraction)
        message = rng.integers(
            0, 2, size=payload_length, dtype=np.uint8
        ).tolist()
        result = embed(
            image,
            message,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
            mapping_tables=tables,
        )
        wire = serialize_auxiliary(result.auxiliary)
        restored, recovered = extract(
            result.stego,
            deserialize_auxiliary(wire, image_shape=result.stego.shape),
        )
        side_bits = auxiliary_bits(result.auxiliary)
        rows.append(
            {
                "image": path.name,
                "foreground_fraction": float(image.mean()),
                "maximum_capacity_bits": maximum,
                "gross_payload_bits": payload_length,
                "auxiliary_bits": side_bits,
                "net_payload_bits": payload_length - side_bits,
                "drd": drd(image, result.stego),
                "psnr_db": psnr(image, result.stego),
                "reversible": bool(np.array_equal(restored, image)),
                "message_exact": recovered == message,
            }
        )

    net = np.asarray([row["net_payload_bits"] for row in rows], dtype=float)
    summary = {
        "dataset": str(args.dataset),
        "binarization_threshold": args.threshold,
        "sample_size": len(rows),
        "sample_seed": args.seed,
        "payload_fraction": args.payload_fraction,
        "net_table_optimization": not args.no_net_optimize,
        "images": rows,
        "mean_gross_payload_bits": float(
            np.mean([row["gross_payload_bits"] for row in rows])
        ),
        "mean_auxiliary_bits": float(
            np.mean([row["auxiliary_bits"] for row in rows])
        ),
        "mean_net_payload_bits": float(net.mean()),
        "median_net_payload_bits": float(np.median(net)),
        "positive_net_images": int(np.count_nonzero(net > 0)),
        "positive_net_fraction": float(np.mean(net > 0)),
        "mean_drd": float(np.mean([row["drd"] for row in rows])),
        "all_reversible": all(row["reversible"] for row in rows),
        "all_messages_exact": all(row["message_exact"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
