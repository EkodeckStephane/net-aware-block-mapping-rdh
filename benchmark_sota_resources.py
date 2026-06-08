"""Benchmark wall time and peak RSS for the four common-protocol methods."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
from time import perf_counter, sleep
import zlib

import numpy as np
import psutil

from abm_rdh import (
    build_mapping_tables,
    deserialize_auxiliary as deserialize_abm,
    drd,
    embed as embed_abm,
    extract as extract_abm,
    serialize_auxiliary as serialize_abm,
)
from dong_adaptive import (
    deserialize_auxiliary as deserialize_dong,
    embed as embed_dong,
    extract as extract_dong,
    serialize_auxiliary as serialize_dong,
)
from evaluate_bossbase_net import DEFAULT_BOSSBASE, load_thresholded
from evaluate_ppocp_comparison import _selected_paths
from evaluate_sota_baselines import _abm_tables_for_payload
from huynh_nguyen import (
    deserialize_auxiliary as deserialize_huynh,
    embed as embed_huynh,
    extract as extract_huynh,
    serialize_auxiliary as serialize_huynh,
)
from ml_candidate_ranker import CandidateRanker
from ppocp import (
    deserialize_auxiliary as deserialize_ppocp,
    embed as embed_ppocp,
    extract as extract_ppocp,
    fit_profile,
    serialize_auxiliary as serialize_ppocp,
)


def measure(function) -> tuple[float, int, object]:
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    peak = baseline
    running = True

    def sample() -> None:
        nonlocal peak
        while running:
            peak = max(peak, process.memory_info().rss)
            sleep(0.002)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = perf_counter()
    try:
        result = function()
    finally:
        elapsed = perf_counter() - started
        running = False
        thread.join()
        peak = max(peak, process.memory_info().rss)
    return elapsed, max(0, peak - baseline), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_BOSSBASE)
    parser.add_argument("--payload", type=int, default=256)
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=Path("ml_ranker_results/candidate_ranker.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sota_multiload/resource_benchmark.json"),
    )
    args = parser.parse_args()

    paths = sorted(args.dataset.glob("*.pgm"), key=lambda path: int(path.stem))
    train, test = _selected_paths(
        paths, train_size=100, test_size=100, seed=args.seed
    )
    profile = fit_profile(load_thresholded(path, 128) for path in train)
    ranker = CandidateRanker.load(args.ranker_model)
    rows = []
    for image_index, path in enumerate(test[: args.images]):
        image = load_thresholded(path, 128)
        message = np.random.default_rng(args.seed + image_index).integers(
            0, 2, size=args.payload, dtype=np.uint8
        ).tolist()

        def abm():
            tables = _abm_tables_for_payload(
                image,
                build_mapping_tables(
                    image, policy="cnn_hamming1", candidate_ranker=ranker
                ),
                args.payload,
            )
            result = embed_abm(
                image,
                message,
                policy="cnn_hamming1",
                candidate_ranker=ranker,
                mapping_tables=tables,
            )
            return extract_abm(
                result.stego,
                deserialize_abm(serialize_abm(result.auxiliary), image_shape=image.shape),
            )

        def ppocp():
            result = embed_ppocp(image, message, profile)
            return extract_ppocp(
                result.stego, deserialize_ppocp(serialize_ppocp(result.auxiliary))
            )

        def huynh():
            result = embed_huynh(image, message)
            return extract_huynh(
                result.stego, deserialize_huynh(serialize_huynh(result.auxiliary))
            )

        def dong():
            candidates = []
            for divisor in range(1, 11):
                try:
                    result = embed_dong(image, message, context_divisor=divisor)
                    restored, recovered = extract_dong(
                        result.stego,
                        deserialize_dong(serialize_dong(result.auxiliary)),
                    )
                    if np.array_equal(restored, image) and recovered == message:
                        candidates.append((drd(image, result.stego), result))
                except (ValueError, zlib.error):
                    pass
            if not candidates:
                raise ValueError("Dong unavailable")
            return min(candidates, key=lambda item: item[0])[1]

        for method, function in (
            ("ABM-enumerative", abm),
            ("PPOCP-conservative", ppocp),
            ("Huynh-Nguyen-T5", huynh),
            ("Dong-adaptive-conservative", dong),
        ):
            try:
                elapsed, peak_rss, _ = measure(function)
                rows.append(
                    {
                        "image": path.name,
                        "method": method,
                        "runtime_seconds": elapsed,
                        "peak_rss_increase_bytes": peak_rss,
                        "available": True,
                    }
                )
            except ValueError as error:
                rows.append(
                    {
                        "image": path.name,
                        "method": method,
                        "runtime_seconds": None,
                        "peak_rss_increase_bytes": None,
                        "available": False,
                        "error": str(error),
                    }
                )

    summaries = []
    for method in sorted({row["method"] for row in rows}):
        selected = [
            row for row in rows if row["method"] == method and row["available"]
        ]
        summaries.append(
            {
                "method": method,
                "available_images": len(selected),
                "median_runtime_seconds": float(
                    np.median([row["runtime_seconds"] for row in selected])
                ),
                "median_peak_rss_increase_mib": float(
                    np.median(
                        [row["peak_rss_increase_bytes"] for row in selected]
                    )
                    / (1024**2)
                ),
                "maximum_peak_rss_increase_mib": float(
                    np.max([row["peak_rss_increase_bytes"] for row in selected])
                    / (1024**2)
                ),
            }
        )
    report = {
        "payload_bits": args.payload,
        "images_requested": args.images,
        "memory_definition": (
            "Peak process RSS minus pre-call RSS, sampled every 2 ms."
        ),
        "summaries": summaries,
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
