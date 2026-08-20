"""Frozen external validation of ABM on the FUNSD scanned-document corpus.

This protocol does not train or tune the CNN on FUNSD.  It uses the committed
candidate-ranker weights and the deployed cnn_hamming1 policy.  Two activation
variants are compared under identical messages:

A1: all feasible distance-one mapping tables, with the exact wire charged.
A2: serialization-aware prefix selected by map-level net capacity.

The primary binarization is a fixed threshold of 128, matching the BOSSbase
transfer protocol in the manuscript.  Otsu can be run as a secondary
preprocessing sensitivity check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

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


PAYLOADS = (64, 128, 256, 512)
SEED = 20260608


def discover_images(root: Path) -> list[tuple[str, Path]]:
    """Return all official FUNSD train/test images with explicit split labels."""
    candidates: list[tuple[str, Path]] = []
    for split_name, relative in (
        ("train", Path("training_data/images")),
        ("test", Path("testing_data/images")),
    ):
        directory = root / relative
        for path in sorted(directory.glob("*.png")):
            candidates.append((split_name, path))
    return candidates


def load_binary(path: Path, mode: str) -> tuple[np.ndarray, int]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if mode == "fixed128":
        threshold = 128
    elif mode == "otsu":
        threshold = int(round(float(threshold_otsu(gray))))
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")
    binary = (gray >= threshold).astype(np.uint8)
    return binary, threshold


def message_for(image_name: str, payload: int) -> list[int]:
    token = f"FUNSD|{image_name}|{payload}|{SEED}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=payload, dtype=np.uint8).tolist()


def evaluate_variant(
    image: np.ndarray,
    image_name: str,
    payload: int,
    variant: str,
    tables: dict[int, dict[int, int]],
    ranker: CandidateRanker,
) -> dict[str, Any]:
    maximum = int(capacity_bits(image, tables))
    base: dict[str, Any] = {
        "variant": variant,
        "target_payload_bits": payload,
        "gross_mapping_capacity_bits": maximum,
        "active_tables": len(tables),
        "available": False,
        "auxiliary_bits": None,
        "net_payload_bits": None,
        "drd": None,
        "psnr_db": None,
        "changed_pixels": None,
        "message_exact": False,
        "cover_exact": False,
        "error": "",
    }
    if payload > maximum:
        base["error"] = "insufficient_mapping_capacity"
        return base

    message = message_for(image_name, payload)
    try:
        result = embed(
            image,
            message,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
            mapping_tables=tables,
        )
        wire = serialize_auxiliary(result.auxiliary)
        decoded_aux = deserialize_auxiliary(wire, image_shape=result.stego.shape)
        restored, recovered = extract(result.stego, decoded_aux)
        side_bits = int(auxiliary_bits(result.auxiliary))
        message_exact = recovered == message
        cover_exact = bool(np.array_equal(restored, image))
        available = bool(message_exact and cover_exact)
        base.update(
            {
                "available": available,
                "auxiliary_bits": side_bits,
                "net_payload_bits": int(payload - side_bits),
                "drd": float(drd(image, result.stego)),
                "psnr_db": float(psnr(image, result.stego)),
                "changed_pixels": int(np.count_nonzero(result.stego != image)),
                "message_exact": bool(message_exact),
                "cover_exact": bool(cover_exact),
                "error": "" if available else "round_trip_mismatch",
            }
        )
    except Exception as exc:  # preserve failures as auditable rows
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def summarize(rows: list[dict[str, Any]], image_count: int, mode: str) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for variant in ("A1_all_feasible", "A2_serialization_aware"):
        for payload in PAYLOADS:
            subset = [
                row
                for row in rows
                if row["variant"] == variant and row["target_payload_bits"] == payload
            ]
            available = [row for row in subset if row["available"]]
            nets = [float(row["net_payload_bits"]) for row in available]
            aux = [float(row["auxiliary_bits"]) for row in available]
            drds = [float(row["drd"]) for row in available]
            caps = [float(row["gross_mapping_capacity_bits"]) for row in subset]
            groups.append(
                {
                    "variant": variant,
                    "payload_bits": payload,
                    "images": len(subset),
                    "available": len(available),
                    "availability": len(available) / max(len(subset), 1),
                    "mean_gross_mapping_capacity_bits": mean_or_none(caps),
                    "mean_auxiliary_bits": mean_or_none(aux),
                    "mean_net_payload_bits": mean_or_none(nets),
                    "median_net_payload_bits": median_or_none(nets),
                    "positive_net_available": int(sum(value > 0 for value in nets)),
                    "positive_net_fraction_available": (
                        float(np.mean(np.asarray(nets) > 0)) if nets else None
                    ),
                    "mean_drd": mean_or_none(drds),
                    "all_available_round_trips_exact": all(
                        row["message_exact"] and row["cover_exact"] for row in available
                    ),
                }
            )

    paired: list[dict[str, Any]] = []
    index = {
        (row["image"], row["target_payload_bits"], row["variant"]): row for row in rows
    }
    image_names = sorted({row["image"] for row in rows})
    for payload in PAYLOADS:
        differences = []
        aux_savings = []
        drd_differences = []
        for name in image_names:
            a1 = index.get((name, payload, "A1_all_feasible"))
            a2 = index.get((name, payload, "A2_serialization_aware"))
            if not a1 or not a2 or not a1["available"] or not a2["available"]:
                continue
            differences.append(float(a2["net_payload_bits"] - a1["net_payload_bits"]))
            aux_savings.append(float(a1["auxiliary_bits"] - a2["auxiliary_bits"]))
            drd_differences.append(float(a2["drd"] - a1["drd"]))
        paired.append(
            {
                "payload_bits": payload,
                "paired_images": len(differences),
                "mean_net_gain_A2_minus_A1_bits": mean_or_none(differences),
                "median_net_gain_A2_minus_A1_bits": median_or_none(differences),
                "mean_aux_saving_A1_minus_A2_bits": mean_or_none(aux_savings),
                "mean_drd_difference_A2_minus_A1": mean_or_none(drd_differences),
                "fraction_A2_net_not_worse": (
                    float(np.mean(np.asarray(differences) >= 0)) if differences else None
                ),
            }
        )

    return {
        "dataset": "FUNSD",
        "dataset_role": "external frozen real scanned-document validation",
        "dataset_source": "https://guillaumejaume.github.io/FUNSD/dataset.zip",
        "threshold_mode": mode,
        "image_count": image_count,
        "expected_image_count": 199,
        "payloads": list(PAYLOADS),
        "seed": SEED,
        "ranker_model": "ml_ranker_results/candidate_ranker.pt",
        "training_or_tuning_on_funsd": False,
        "group_summary": groups,
        "paired_activation_summary": paired,
        "all_successful_round_trips_exact": all(
            row["message_exact"] and row["cover_exact"]
            for row in rows
            if row["available"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=Path("ml_ranker_results/candidate_ranker.pt"),
    )
    parser.add_argument(
        "--threshold-mode", choices=("fixed128", "otsu"), default="fixed128"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/funsd_external"))
    args = parser.parse_args()

    images = discover_images(args.dataset)
    if len(images) != 199:
        raise RuntimeError(f"Expected 199 FUNSD images, found {len(images)} under {args.dataset}")

    ranker = CandidateRanker.load(args.ranker_model)
    rows: list[dict[str, Any]] = []
    for index, (split, path) in enumerate(images, start=1):
        image, threshold = load_binary(path, args.threshold_mode)
        tables_a1 = build_mapping_tables(
            image,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        )
        tables_a2 = optimize_tables_for_net_capacity(
            image,
            tables_a1,
            policy="cnn_hamming1",
        )
        common = {
            "image": path.name,
            "split": split,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "threshold_mode": args.threshold_mode,
            "threshold_value": threshold,
            "one_fraction": float(image.mean()),
            "black_fraction": float(1.0 - image.mean()),
        }
        for payload in PAYLOADS:
            for variant, tables in (
                ("A1_all_feasible", tables_a1),
                ("A2_serialization_aware", tables_a2),
            ):
                row = dict(common)
                row.update(
                    evaluate_variant(
                        image,
                        path.name,
                        payload,
                        variant,
                        tables,
                        ranker,
                    )
                )
                rows.append(row)
        print(f"[{index:03d}/199] {split}/{path.name}: A1={len(tables_a1)} tables, A2={len(tables_a2)} tables")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "funsd_external_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, len(images), args.threshold_mode)
    summary_path = args.output_dir / "funsd_external_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    paired_path = args.output_dir / "funsd_activation_paired.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        paired_rows = summary["paired_activation_summary"]
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
