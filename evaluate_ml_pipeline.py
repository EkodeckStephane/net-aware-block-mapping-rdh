"""Evaluate constrained RDH plus RF-guided uniform-block embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
from ml_uniform_agent import UniformBlockAgent
from ml_uniform_embedding import (
    deserialize_uniform_auxiliary,
    embed_uniform_bits,
    extract_uniform_bits,
    serialize_uniform_auxiliary,
    uniform_auxiliary_bits,
    uniform_capacity,
)
from run_experiments import DEFAULT_IMAGES, load_binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument(
        "--uniform-model",
        type=Path,
        default=Path("ml_uniform_results/uniform_agent.joblib"),
    )
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=Path("ml_ranker_results/candidate_ranker.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("ml_pipeline_results.json"))
    parser.add_argument("--base-payload-fraction", type=float, default=0.75)
    parser.add_argument("--uniform-payload-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    agent = UniformBlockAgent.load(args.uniform_model)
    ranker = CandidateRanker.load(args.ranker_model)
    rng = np.random.default_rng(args.seed)
    rows = []
    for filename in DEFAULT_IMAGES:
        image = load_binary(args.image_dir / filename)
        base_tables = build_mapping_tables(
            image,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        )
        base_tables = optimize_tables_for_net_capacity(
            image,
            base_tables,
            policy="cnn_hamming1",
        )
        base_capacity = capacity_bits(image, base_tables)
        base_payload = int(base_capacity * args.base_payload_fraction)
        base_message = rng.integers(
            0, 2, size=base_payload, dtype=np.uint8
        ).tolist()
        base_result = embed(
            image,
            base_message,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
            mapping_tables=base_tables,
        )

        uniform_maximum = uniform_capacity(base_result.stego, agent)
        uniform_payload = int(uniform_maximum * args.uniform_payload_fraction)
        uniform_message = rng.integers(
            0, 2, size=uniform_payload, dtype=np.uint8
        ).tolist()
        combined_stego, uniform_auxiliary = embed_uniform_bits(
            base_result.stego,
            uniform_message,
            agent,
        )

        uniform_wire = serialize_uniform_auxiliary(uniform_auxiliary)
        base_wire = serialize_auxiliary(base_result.auxiliary)
        restored_base_stego, recovered_uniform = extract_uniform_bits(
            combined_stego,
            deserialize_uniform_auxiliary(uniform_wire),
            agent,
        )
        restored, recovered_base = extract(
            restored_base_stego,
            deserialize_auxiliary(base_wire, image_shape=restored_base_stego.shape),
        )
        base_side_bits = auxiliary_bits(base_result.auxiliary)
        uniform_side_bits = uniform_auxiliary_bits(uniform_auxiliary)
        gross_payload = base_payload + uniform_payload
        net_payload = gross_payload - base_side_bits - uniform_side_bits
        rows.append(
            {
                "image": Path(filename).stem,
                "base_payload_bits": base_payload,
                "uniform_payload_bits": uniform_payload,
                "uniform_maximum_bits": uniform_maximum,
                "gross_payload_bits": gross_payload,
                "base_auxiliary_bits": base_side_bits,
                "uniform_auxiliary_bits": uniform_side_bits,
                "total_auxiliary_bits": base_side_bits + uniform_side_bits,
                "net_payload_bits": net_payload,
                "base_net_payload_bits": base_payload - base_side_bits,
                "uniform_incremental_net_bits": uniform_payload
                - uniform_side_bits,
                "uniform_gain_percent": 100.0
                * uniform_payload
                / max(base_payload, 1),
                "base_drd": drd(image, base_result.stego),
                "combined_drd": drd(image, combined_stego),
                "combined_psnr_db": psnr(image, combined_stego),
                "reversible": bool(np.array_equal(restored, image)),
                "base_message_exact": recovered_base == base_message,
                "uniform_message_exact": recovered_uniform == uniform_message,
            }
        )

    summary = {
        "images": rows,
        "mean_uniform_payload_bits": sum(
            row["uniform_payload_bits"] for row in rows
        )
        / len(rows),
        "mean_gross_payload_bits": sum(row["gross_payload_bits"] for row in rows)
        / len(rows),
        "mean_net_payload_bits": sum(row["net_payload_bits"] for row in rows)
        / len(rows),
        "mean_base_net_payload_bits": sum(
            row["base_net_payload_bits"] for row in rows
        )
        / len(rows),
        "mean_uniform_incremental_net_bits": sum(
            row["uniform_incremental_net_bits"] for row in rows
        )
        / len(rows),
        "mean_total_auxiliary_bits": sum(
            row["total_auxiliary_bits"] for row in rows
        )
        / len(rows),
        "mean_uniform_gain_percent": sum(
            row["uniform_gain_percent"] for row in rows
        )
        / len(rows),
        "mean_combined_drd": sum(row["combined_drd"] for row in rows) / len(rows),
        "maximum_combined_drd": max(row["combined_drd"] for row in rows),
        "all_reversible": all(row["reversible"] for row in rows),
        "all_messages_exact": all(
            row["base_message_exact"] and row["uniform_message_exact"] for row in rows
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
