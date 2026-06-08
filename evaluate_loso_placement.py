"""Evaluate CNN candidate ordering with leave-one-image-out training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from abm_rdh import build_mapping_tables, capacity_bits, evaluate
from ml_candidate_ranker import CandidateCostCNN, CandidateRanker, ranker_dataset
from run_experiments import DEFAULT_IMAGES, DOCUMENT_IMAGE_DIR, load_binary


def train_fold_ranker(
    training_images: list[tuple[str, np.ndarray]],
    *,
    epochs: int,
    seed: int,
) -> CandidateRanker:
    torch.manual_seed(seed)
    tensors, targets, _ = ranker_dataset(training_images)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tensors), torch.from_numpy(targets)),
        batch_size=128,
        shuffle=True,
    )
    model = CandidateCostCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_function = torch.nn.SmoothL1Loss()
    for _ in range(epochs):
        model.train()
        for features, target in loader:
            prediction = model(features)
            loss = loss_function(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return CandidateRanker(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=DOCUMENT_IMAGE_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--payload-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml_ranker_results/loso_placement.json"),
    )
    args = parser.parse_args()

    named_images = [
        (Path(filename).stem, load_binary(args.image_dir / filename))
        for filename in DEFAULT_IMAGES
    ]
    rows: list[dict[str, object]] = []
    for fold, (held_out_name, held_out_image) in enumerate(named_images):
        training_images = [
            item for index, item in enumerate(named_images) if index != fold
        ]
        ranker = train_fold_ranker(
            training_images,
            epochs=args.epochs,
            seed=args.seed + fold,
        )
        baseline_tables = build_mapping_tables(held_out_image, policy="hamming1")
        cnn_tables = build_mapping_tables(
            held_out_image,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        )
        maximum = min(
            capacity_bits(held_out_image, baseline_tables),
            capacity_bits(held_out_image, cnn_tables),
        )
        payload = int(maximum * args.payload_fraction)
        rng = np.random.default_rng(args.seed + fold)
        message = rng.integers(0, 2, size=payload, dtype=np.uint8).tolist()
        _, _, _, baseline = evaluate(
            held_out_image,
            message,
            policy="hamming1",
        )
        _, _, _, cnn = evaluate(
            held_out_image,
            message,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        )
        baseline_drd = float(baseline["drd"])
        cnn_drd = float(cnn["drd"])
        rows.append(
            {
                "image": held_out_name,
                "payload_bits": payload,
                "baseline_drd": baseline_drd,
                "cnn_drd": cnn_drd,
                "drd_reduction_percent": (
                    100.0 * (baseline_drd - cnn_drd) / baseline_drd
                    if baseline_drd
                    else 0.0
                ),
                "reversible": bool(cnn["reversible"]),
                "message_exact": bool(cnn["message_exact"]),
            }
        )

    baseline_mean = float(np.mean([row["baseline_drd"] for row in rows]))
    cnn_mean = float(np.mean([row["cnn_drd"] for row in rows]))
    reductions = [float(row["drd_reduction_percent"]) for row in rows]
    report = {
        "protocol": "leave-one-image-out placement evaluation",
        "epochs": args.epochs,
        "payload_fraction": args.payload_fraction,
        "folds": rows,
        "macro_baseline_drd": baseline_mean,
        "macro_cnn_drd": cnn_mean,
        "macro_drd_reduction_percent": (
            100.0 * (baseline_mean - cnn_mean) / baseline_mean
            if baseline_mean
            else 0.0
        ),
        "mean_per_image_reduction_percent": float(np.mean(reductions)),
        "all_reversible": all(bool(row["reversible"]) for row in rows),
        "all_messages_exact": all(bool(row["message_exact"]) for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
