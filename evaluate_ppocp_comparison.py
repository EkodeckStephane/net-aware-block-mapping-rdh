"""Matched-payload comparison between PPOCP and the proposed ABM pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from abm_rdh import (
    auxiliary_bits as abm_auxiliary_bits,
    build_mapping_tables,
    capacity_bits as abm_capacity_bits,
    deserialize_auxiliary as deserialize_abm_auxiliary,
    drd,
    embed as embed_abm,
    extract as extract_abm,
    optimize_tables_for_net_capacity,
    psnr,
    serialize_auxiliary as serialize_abm_auxiliary,
)
from evaluate_bossbase_net import DEFAULT_BOSSBASE, load_thresholded
from ml_candidate_ranker import CandidateRanker
from ppocp import (
    auxiliary_bits as ppocp_auxiliary_bits,
    deserialize_auxiliary as deserialize_ppocp_auxiliary,
    embed as embed_ppocp,
    extract as extract_ppocp,
    fit_profile,
    pair_positions,
    serialize_auxiliary as serialize_ppocp_auxiliary,
)


def _selected_paths(
    paths: list[Path],
    *,
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if train_size + test_size > len(paths):
        raise ValueError("Requested train and test sets exceed the dataset")
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(paths),
        size=train_size + test_size,
        replace=False,
    ).tolist()
    train = [paths[index] for index in sorted(indices[:train_size])]
    test = [paths[index] for index in sorted(indices[train_size:])]
    return train, test


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    fractions = sorted({float(row["payload_fraction"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    for fraction in fractions:
        for method in methods:
            selected = [
                row
                for row in rows
                if float(row["payload_fraction"]) == fraction
                and str(row["method"]) == method
            ]
            net = np.asarray([float(row["net_payload_bits"]) for row in selected])
            psnr_values = np.asarray(
                [float(row["psnr_db"]) for row in selected],
                dtype=float,
            )
            finite_psnr = psnr_values[np.isfinite(psnr_values)]
            summaries.append(
                {
                    "method": method,
                    "payload_fraction": fraction,
                    "images": len(selected),
                    "mean_gross_payload_bits": float(
                        np.mean([float(row["gross_payload_bits"]) for row in selected])
                    ),
                    "mean_auxiliary_bits": float(
                        np.mean([float(row["auxiliary_bits"]) for row in selected])
                    ),
                    "mean_net_payload_bits": float(net.mean()),
                    "median_net_payload_bits": float(np.median(net)),
                    "positive_net_fraction": float(np.mean(net > 0)),
                    "mean_drd": float(
                        np.mean([float(row["drd"]) for row in selected])
                    ),
                    "mean_psnr_db_finite": (
                        float(finite_psnr.mean()) if finite_psnr.size else None
                    ),
                    "infinite_psnr_images": int(
                        np.count_nonzero(~np.isfinite(psnr_values))
                    ),
                    "all_reversible": all(bool(row["reversible"]) for row in selected),
                    "all_messages_exact": all(
                        bool(row["message_exact"]) for row in selected
                    ),
                }
            )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_BOSSBASE)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=Path("ml_ranker_results/candidate_ranker.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ppocp_comparison_results.json"),
    )
    args = parser.parse_args()

    paths = sorted(args.dataset.glob("*.pgm"), key=lambda path: int(path.stem))
    train_paths, test_paths = _selected_paths(
        paths,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    profile = fit_profile(
        load_thresholded(path, args.threshold) for path in train_paths
    )
    ranker = CandidateRanker.load(args.ranker_model)
    rng = np.random.default_rng(args.seed + 1)
    rows: list[dict[str, object]] = []

    for path in test_paths:
        image = load_thresholded(path, args.threshold)
        tables = build_mapping_tables(
            image,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        )
        tables = optimize_tables_for_net_capacity(
            image,
            tables,
            policy="cnn_hamming1",
        )
        abm_maximum = abm_capacity_bits(image, tables)
        ppocp_maximum = max(
            int(pair_positions(image, first_level).size)
            for first_level in range(1, 14)
        )
        common_maximum = min(abm_maximum, ppocp_maximum)

        for fraction in args.fractions:
            payload_length = int(common_maximum * fraction)
            message = rng.integers(
                0,
                2,
                size=payload_length,
                dtype=np.uint8,
            ).tolist()

            abm_result = embed_abm(
                image,
                message,
                policy="cnn_hamming1",
                candidate_ranker=ranker,
                mapping_tables=tables,
            )
            abm_wire = serialize_abm_auxiliary(abm_result.auxiliary)
            abm_restored, abm_recovered = extract_abm(
                abm_result.stego,
                deserialize_abm_auxiliary(abm_wire, image_shape=image.shape),
            )
            rows.append(
                {
                    "image": path.name,
                    "method": "ABM-CNN-H1",
                    "payload_fraction": fraction,
                    "common_maximum_bits": common_maximum,
                    "gross_payload_bits": payload_length,
                    "auxiliary_bits": abm_auxiliary_bits(abm_result.auxiliary),
                    "net_payload_bits": (
                        payload_length - abm_auxiliary_bits(abm_result.auxiliary)
                    ),
                    "changed_pixels": int(
                        np.count_nonzero(image != abm_result.stego)
                    ),
                    "drd": drd(image, abm_result.stego),
                    "psnr_db": psnr(image, abm_result.stego),
                    "reversible": bool(np.array_equal(abm_restored, image)),
                    "message_exact": abm_recovered == message,
                }
            )

            ppocp_result = embed_ppocp(image, message, profile)
            ppocp_wire = serialize_ppocp_auxiliary(ppocp_result.auxiliary)
            ppocp_restored, ppocp_recovered = extract_ppocp(
                ppocp_result.stego,
                deserialize_ppocp_auxiliary(ppocp_wire),
            )
            rows.append(
                {
                    "image": path.name,
                    "method": "PPOCP-conservative",
                    "payload_fraction": fraction,
                    "common_maximum_bits": common_maximum,
                    "gross_payload_bits": payload_length,
                    "auxiliary_bits": ppocp_auxiliary_bits(
                        ppocp_result.auxiliary
                    ),
                    "net_payload_bits": (
                        payload_length
                        - ppocp_auxiliary_bits(ppocp_result.auxiliary)
                    ),
                    "changed_pixels": ppocp_result.changed_pixels,
                    "drd": drd(image, ppocp_result.stego),
                    "psnr_db": psnr(image, ppocp_result.stego),
                    "reversible": bool(np.array_equal(ppocp_restored, image)),
                    "message_exact": ppocp_recovered == message,
                }
            )


    report = {
        "dataset": str(args.dataset),
        "threshold": args.threshold,
        "seed": args.seed,
        "train_size": len(train_paths),
        "test_size": len(test_paths),
        "train_test_disjoint": not bool(set(train_paths) & set(test_paths)),
        "ppocp_assumption": (
            "Used overlapping-block centers and original center values are "
            "serialized to guarantee exact synchronization and restoration."
        ),
        "profile_single_flip_cost": list(profile.single_flip_cost),
        "summaries": _summaries(rows),
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
