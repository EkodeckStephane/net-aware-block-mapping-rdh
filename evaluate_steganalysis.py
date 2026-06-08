"""Grouped cover/stego detection for the common-protocol embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from abm_rdh import build_mapping_tables, embed as embed_abm
from dong_adaptive import embed as embed_dong
from evaluate_bossbase_net import load_thresholded
from evaluate_ppocp_comparison import _selected_paths
from evaluate_sota_baselines import _abm_tables_for_payload
from huynh_nguyen import embed as embed_huynh
from ml_candidate_ranker import CandidateRanker
from ppocp import embed as embed_ppocp, fit_profile


def features(image: np.ndarray) -> np.ndarray:
    binary = image.astype(np.uint8)
    values = [
        float(binary.mean()),
        float(np.mean(binary[:, 1:] != binary[:, :-1])),
        float(np.mean(binary[1:, :] != binary[:-1, :])),
    ]
    for size in (2, 3):
        codes = np.zeros(
            (binary.shape[0] - size + 1, binary.shape[1] - size + 1),
            dtype=np.uint16,
        )
        shift = 0
        for row in range(size):
            for col in range(size):
                codes |= binary[
                    row : row + codes.shape[0], col : col + codes.shape[1]
                ].astype(np.uint16) << shift
                shift += 1
        histogram = np.bincount(codes.ravel(), minlength=2 ** (size * size))
        values.extend((histogram / histogram.sum()).tolist())
    return np.asarray(values, dtype=np.float64)


def score_detector(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, model
) -> dict[str, float]:
    folds = min(5, len(np.unique(groups)))
    probabilities = cross_val_predict(
        model,
        x,
        y,
        groups=groups,
        cv=GroupKFold(folds),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    predictions = probabilities >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "error_probability": float(1.0 - balanced_accuracy_score(y, predictions)),
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=Path("ml_ranker_results/candidate_ranker.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sota_multiload/steganalysis.json"),
    )
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    all_paths = sorted(args.dataset.glob("*.pgm"), key=lambda path: int(path.stem))
    train_paths, test_paths = _selected_paths(
        all_paths, train_size=100, test_size=100, seed=args.seed
    )
    test_index = {path.name: index for index, path in enumerate(test_paths)}
    profile = fit_profile(load_thresholded(path, 128) for path in train_paths)
    ranker = CandidateRanker.load(args.ranker_model)
    results = []

    for report_path in args.reports:
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        payload = int(report["payload_bits"])
        rows = report["images"]
        by_method = {
            method: {
                row["image"]: row
                for row in rows
                if row["method"] == method and row["available"]
            }
            for method in (
                "ABM-enumerative",
                "Dong-adaptive-conservative",
                "Huynh-Nguyen-T5",
                "PPOCP-conservative",
            )
        }
        common = sorted(set.intersection(*(set(items) for items in by_method.values())))
        for method, method_rows in by_method.items():
            samples = []
            labels = []
            groups = []
            for group_index, name in enumerate(common):
                image = load_thresholded(args.dataset / name, 128)
                message = np.random.default_rng(args.seed + test_index[name]).integers(
                    0, 2, size=payload, dtype=np.uint8
                ).tolist()
                if method == "ABM-enumerative":
                    tables = _abm_tables_for_payload(
                        image,
                        build_mapping_tables(
                            image,
                            policy="cnn_hamming1",
                            candidate_ranker=ranker,
                        ),
                        payload,
                    )
                    stego = embed_abm(
                        image,
                        message,
                        policy="cnn_hamming1",
                        candidate_ranker=ranker,
                        mapping_tables=tables,
                    ).stego
                elif method == "Dong-adaptive-conservative":
                    divisor = int(method_rows[name]["parameter"].split("=")[1])
                    stego = embed_dong(
                        image, message, context_divisor=divisor
                    ).stego
                elif method == "Huynh-Nguyen-T5":
                    stego = embed_huynh(image, message).stego
                else:
                    stego = embed_ppocp(image, message, profile).stego
                samples.extend((features(image), features(stego)))
                labels.extend((0, 1))
                groups.extend((group_index, group_index))
            x = np.vstack(samples)
            y = np.asarray(labels)
            group_array = np.asarray(groups)
            classifiers = {
                "logistic": make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, random_state=args.seed),
                ),
                "random_forest": RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            }
            for classifier, model in classifiers.items():
                results.append(
                    {
                        "payload_bits": payload,
                        "method": method,
                        "classifier": classifier,
                        "paired_images": len(common),
                        **score_detector(x, y, group_array, model),
                    }
                )

    report = {
        "feature_definition": (
            "Foreground and transition rates plus normalized 2x2 and 3x3 "
            "binary pattern histograms."
        ),
        "validation": (
            "Five-fold GroupKFold; each cover and its stego remain in the same fold."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
