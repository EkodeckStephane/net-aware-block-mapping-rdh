"""Common-message BOSSbase comparison of ABM, PPOCP, Huynh, and Dong."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import zlib

import numpy as np

from abm_rdh import (
    auxiliary_bits as abm_auxiliary_bits,
    build_mapping_tables,
    capacity_bits as abm_capacity_bits,
    deserialize_auxiliary as deserialize_abm_auxiliary,
    drd,
    embed as embed_abm,
    extract as extract_abm,
    iter_blocks,
    pattern_index,
    psnr,
    serialize_auxiliary as serialize_abm_auxiliary,
)
from dong_adaptive import (
    auxiliary_bits as dong_auxiliary_bits,
    deserialize_auxiliary as deserialize_dong_auxiliary,
    embed as embed_dong,
    extract as extract_dong,
    serialize_auxiliary as serialize_dong_auxiliary,
)
from evaluate_bossbase_net import DEFAULT_BOSSBASE, load_thresholded
from evaluate_ppocp_comparison import _selected_paths
from huynh_nguyen import (
    auxiliary_bits as huynh_auxiliary_bits,
    capacity_bits as huynh_capacity_bits,
    deserialize_auxiliary as deserialize_huynh_auxiliary,
    embed as embed_huynh,
    extract as extract_huynh,
    serialize_auxiliary as serialize_huynh_auxiliary,
)
from ml_candidate_ranker import CandidateRanker
from ppocp import (
    auxiliary_bits as ppocp_auxiliary_bits,
    deserialize_auxiliary as deserialize_ppocp_auxiliary,
    embed as embed_ppocp,
    extract as extract_ppocp,
    fit_profile,
    serialize_auxiliary as serialize_ppocp_auxiliary,
)


def _row(
    *,
    image_name: str,
    method: str,
    payload: int,
    auxiliary: int,
    original: np.ndarray,
    stego: np.ndarray,
    reversible: bool,
    message_exact: bool,
    parameter: str = "",
    runtime_seconds: float | None = None,
) -> dict[str, object]:
    return {
        "image": image_name,
        "method": method,
        "available": True,
        "payload_bits": payload,
        "auxiliary_bits": auxiliary,
        "net_payload_bits": payload - auxiliary,
        "changed_pixels": int(np.count_nonzero(original != stego)),
        "drd": drd(original, stego),
        "psnr_db": psnr(original, stego),
        "reversible": reversible,
        "message_exact": message_exact,
        "parameter": parameter,
        "runtime_seconds": runtime_seconds,
        "error": "",
    }


def _abm_tables_for_payload(
    image: np.ndarray,
    tables: dict[int, dict[int, int]],
    payload: int,
) -> dict[int, dict[int, int]]:
    frequencies: dict[int, int] = {}
    for _, _, block in iter_blocks(image):
        pattern = pattern_index(block)
        frequencies[pattern] = frequencies.get(pattern, 0) + 1
    selected: dict[int, dict[int, int]] = {}
    capacity = 0
    for peak in sorted(tables, key=lambda value: (-frequencies.get(value, 0), value)):
        selected[peak] = tables[peak]
        capacity += frequencies.get(peak, 0)
        if capacity >= payload:
            return selected
    raise ValueError("capacity below target")


def _unavailable(
    image_name: str,
    method: str,
    payload: int,
    error: str,
    runtime_seconds: float | None = None,
) -> dict[str, object]:
    return {
        "image": image_name,
        "method": method,
        "available": False,
        "payload_bits": payload,
        "auxiliary_bits": 0,
        "net_payload_bits": 0,
        "changed_pixels": 0,
        "drd": None,
        "psnr_db": None,
        "reversible": False,
        "message_exact": False,
        "parameter": "",
        "runtime_seconds": runtime_seconds,
        "error": error,
    }


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        available = [row for row in selected if bool(row["available"])]
        summary: dict[str, object] = {
            "method": method,
            "images": len(selected),
            "available_images": len(available),
            "availability_fraction": len(available) / len(selected),
            "all_available_runs_exact": all(
                bool(row["reversible"]) and bool(row["message_exact"])
                for row in available
            ),
        }
        if available:
            summary.update(
                {
                    "mean_auxiliary_bits": float(
                        np.mean([float(row["auxiliary_bits"]) for row in available])
                    ),
                    "mean_net_payload_bits": float(
                        np.mean([float(row["net_payload_bits"]) for row in available])
                    ),
                    "median_net_payload_bits": float(
                        np.median([float(row["net_payload_bits"]) for row in available])
                    ),
                    "positive_net_fraction": float(
                        np.mean(
                            [float(row["net_payload_bits"]) > 0 for row in available]
                        )
                    ),
                    "mean_changed_pixels": float(
                        np.mean([float(row["changed_pixels"]) for row in available])
                    ),
                    "mean_drd": float(
                        np.mean([float(row["drd"]) for row in available])
                    ),
                    "median_drd": float(
                        np.median([float(row["drd"]) for row in available])
                    ),
                    "maximum_drd": float(
                        np.max([float(row["drd"]) for row in available])
                    ),
                    "drd_at_most_0_5_fraction": float(
                        np.mean([float(row["drd"]) <= 0.5 for row in available])
                    ),
                    "drd_at_most_1_fraction": float(
                        np.mean([float(row["drd"]) <= 1.0 for row in available])
                    ),
                }
            )
        summaries.append(summary)
    available_by_image: dict[str, set[str]] = {}
    for row in rows:
        if row["available"]:
            available_by_image.setdefault(str(row["image"]), set()).add(
                str(row["method"])
            )
    matched_images = {
        image
        for image, present in available_by_image.items()
        if len(present) == len(methods)
    }
    for summary in summaries:
        method_rows = [
            row
            for row in rows
            if row["method"] == summary["method"]
            and row["image"] in matched_images
        ]
        summary["all_method_matched_images"] = len(matched_images)
        if method_rows:
            summary["matched_mean_net_payload_bits"] = float(
                np.mean([float(row["net_payload_bits"]) for row in method_rows])
            )
            summary["matched_mean_drd"] = float(
                np.mean([float(row["drd"]) for row in method_rows])
            )
            summary["matched_median_drd"] = float(
                np.median([float(row["drd"]) for row in method_rows])
            )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_BOSSBASE)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--payload", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=Path("ml_ranker_results/candidate_ranker.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sota_baseline_results.json"),
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
    rows: list[dict[str, object]] = []

    for image_index, path in enumerate(test_paths):
        image = load_thresholded(path, args.threshold)
        message = np.random.default_rng(
            args.seed + image_index
        ).integers(0, 2, size=args.payload, dtype=np.uint8).tolist()

        started = perf_counter()
        try:
            tables = build_mapping_tables(
                image,
                policy="cnn_hamming1",
                candidate_ranker=ranker,
            )
            tables = _abm_tables_for_payload(
                image,
                tables,
                args.payload,
            )
            if abm_capacity_bits(image, tables) < args.payload:
                raise ValueError("capacity below target")
            result = embed_abm(
                image,
                message,
                policy="cnn_hamming1",
                candidate_ranker=ranker,
                mapping_tables=tables,
            )
            wire = serialize_abm_auxiliary(result.auxiliary)
            restored, recovered = extract_abm(
                result.stego,
                deserialize_abm_auxiliary(wire, image_shape=image.shape),
            )
            rows.append(
                _row(
                    image_name=path.name,
                    method="ABM-enumerative",
                    payload=args.payload,
                    auxiliary=abm_auxiliary_bits(result.auxiliary),
                    original=image,
                    stego=result.stego,
                    reversible=bool(np.array_equal(restored, image)),
                    message_exact=recovered == message,
                    runtime_seconds=perf_counter() - started,
                )
            )
        except ValueError as error:
            rows.append(
                _unavailable(
                    path.name,
                    "ABM-enumerative",
                    args.payload,
                    str(error),
                    perf_counter() - started,
                )
            )

        started = perf_counter()
        try:
            result = embed_ppocp(image, message, profile)
            wire = serialize_ppocp_auxiliary(result.auxiliary)
            restored, recovered = extract_ppocp(
                result.stego,
                deserialize_ppocp_auxiliary(wire),
            )
            rows.append(
                _row(
                    image_name=path.name,
                    method="PPOCP-conservative",
                    payload=args.payload,
                    auxiliary=ppocp_auxiliary_bits(result.auxiliary),
                    original=image,
                    stego=result.stego,
                    reversible=bool(np.array_equal(restored, image)),
                    message_exact=recovered == message,
                    runtime_seconds=perf_counter() - started,
                )
            )
        except ValueError as error:
            rows.append(
                _unavailable(
                    path.name,
                    "PPOCP-conservative",
                    args.payload,
                    str(error),
                    perf_counter() - started,
                )
            )

        started = perf_counter()
        try:
            if huynh_capacity_bits(image) < args.payload:
                raise ValueError("capacity below target")
            result = embed_huynh(image, message)
            wire = serialize_huynh_auxiliary(result.auxiliary)
            restored, recovered = extract_huynh(
                result.stego,
                deserialize_huynh_auxiliary(wire),
            )
            rows.append(
                _row(
                    image_name=path.name,
                    method="Huynh-Nguyen-T5",
                    payload=args.payload,
                    auxiliary=huynh_auxiliary_bits(result.auxiliary),
                    original=image,
                    stego=result.stego,
                    reversible=bool(np.array_equal(restored, image)),
                    message_exact=recovered == message,
                    runtime_seconds=perf_counter() - started,
                )
            )
        except ValueError as error:
            rows.append(
                _unavailable(
                    path.name,
                    "Huynh-Nguyen-T5",
                    args.payload,
                    str(error),
                    perf_counter() - started,
                )
            )

        started = perf_counter()
        dong_candidates = []
        dong_errors = []
        for divisor in range(1, 11):
            try:
                result = embed_dong(
                    image,
                    message,
                    context_divisor=divisor,
                )
                wire = serialize_dong_auxiliary(result.auxiliary)
                restored, recovered = extract_dong(
                    result.stego,
                    deserialize_dong_auxiliary(wire),
                )
                if not np.array_equal(restored, image) or recovered != message:
                    raise ValueError("round trip mismatch")
                dong_candidates.append(
                    (
                        drd(image, result.stego),
                        divisor,
                        result,
                    )
                )
            except (ValueError, zlib.error) as error:
                dong_errors.append(f"l={divisor}: {error}")
        if dong_candidates:
            _, divisor, result = min(
                dong_candidates,
                key=lambda item: (item[0], item[1]),
            )
            rows.append(
                _row(
                    image_name=path.name,
                    method="Dong-adaptive-conservative",
                    payload=args.payload,
                    auxiliary=dong_auxiliary_bits(result.auxiliary),
                    original=image,
                    stego=result.stego,
                    reversible=True,
                    message_exact=True,
                    parameter=f"l={divisor}",
                    runtime_seconds=perf_counter() - started,
                )
            )
        else:
            rows.append(
                _unavailable(
                    path.name,
                    "Dong-adaptive-conservative",
                    args.payload,
                    "; ".join(dong_errors),
                    perf_counter() - started,
                )
            )

    report = {
        "dataset": str(args.dataset),
        "threshold": args.threshold,
        "payload_bits": args.payload,
        "seed": args.seed,
        "train_size": len(train_paths),
        "test_size": len(test_paths),
        "train_test_disjoint": not bool(set(train_paths) & set(test_paths)),
        "dong_note": (
            "Adaptive Dong core with compressed PF/PFR location map serialized "
            "as side information; l=1..10 optimized for minimum DRD."
        ),
        "summaries": _summaries(rows),
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with args.output.with_suffix(".csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
