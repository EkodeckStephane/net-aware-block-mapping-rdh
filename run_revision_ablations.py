"""Revision-focused ablations and diagnostic artefacts.

The script keeps the original benchmark code intact and generates the focused
evidence requested during review: serialization-aware activation ablation,
ranker baselines, CNN architecture diagnostics, module timings, payload
footprint diagnostics, and a visual cover/stego/residual comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from math import log2
from pathlib import Path
from time import perf_counter
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from abm_rdh import (
    AuxiliaryData,
    auxiliary_bits,
    build_mapping_tables,
    capacity_bits,
    deserialize_auxiliary,
    drd,
    embed,
    extract,
    hamming,
    iter_blocks,
    optimize_tables_for_net_capacity,
    pattern_array,
    pattern_index,
    serialize_auxiliary,
)
from dong_adaptive import embed as embed_dong
from huynh_nguyen import capacity_bits as huynh_capacity_bits
from huynh_nguyen import embed as embed_huynh
from ml_candidate_ranker import CandidateRanker, empirical_replacement_cost
from ml_uniform_agent import flip_cost_map
from ppocp import embed as embed_ppocp
from ppocp import fit_profile as fit_ppocp_profile
from run_experiments import DEFAULT_IMAGES, DOCUMENT_IMAGE_DIR, load_binary


SEED = 20260608


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _summary(rows: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    grouped = sorted({str(row[group_key]) for row in rows})
    out = []
    for key in grouped:
        selected = [row for row in rows if str(row[group_key]) == key]
        available = [row for row in selected if bool(row["available"])]
        item: dict[str, object] = {
            group_key: key,
            "images": len(selected),
            "available_images": len(available),
            "availability": len(available) / len(selected) if selected else 0.0,
        }
        if available:
            for metric in (
                "gross_capacity_bits",
                "payload_bits",
                "auxiliary_bits",
                "net_payload_bits",
                "posthoc_auxiliary_bits",
                "posthoc_net_payload_bits",
                "drd",
                "changed_pixels",
            ):
                values = [
                    float(row[metric])
                    for row in available
                    if row.get(metric) is not None
                ]
                if values:
                    item[f"mean_{metric}"] = float(np.mean(values))
                    item[f"median_{metric}"] = float(np.median(values))
            item["all_reversible"] = all(bool(row["reversible"]) for row in available)
            item["all_messages_exact"] = all(
                bool(row["message_exact"]) for row in available
            )
        out.append(item)
    return out


def _pattern_positions(image: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    positions: dict[int, list[tuple[int, int]]] = {}
    for row, col, block in iter_blocks(image):
        pattern = pattern_index(block)
        positions.setdefault(pattern, []).append((row, col))
    return positions


class DirectDRDRanker:
    name = "direct_drd_cost"

    def __init__(self) -> None:
        self._cache: dict[int, tuple[dict[int, list[tuple[int, int]]], np.ndarray]] = {}

    def predict_costs(
        self,
        image: np.ndarray,
        peak: int,
        candidates: list[int],
    ) -> np.ndarray:
        key = id(image)
        if key not in self._cache:
            self._cache[key] = (_pattern_positions(image), flip_cost_map(image))
        positions, costs = self._cache[key]
        return np.asarray(
            [
                empirical_replacement_cost(
                    image,
                    positions.get(peak, []),
                    peak,
                    candidate,
                    costs,
                )
                for candidate in candidates
            ],
            dtype=np.float64,
        )


class TransitionRanker:
    name = "transition_preserving"

    @staticmethod
    def _transitions(pattern: int) -> int:
        array = pattern_array(pattern)
        return int(np.count_nonzero(array[:, 1:] != array[:, :-1])) + int(
            np.count_nonzero(array[1:, :] != array[:-1, :])
        )

    def predict_costs(
        self,
        image: np.ndarray,
        peak: int,
        candidates: list[int],
    ) -> np.ndarray:
        peak_transitions = self._transitions(peak)
        return np.asarray(
            [
                abs(self._transitions(candidate) - peak_transitions)
                + 0.01 * candidate
                for candidate in candidates
            ],
            dtype=np.float64,
        )


class ConnectivityRanker:
    name = "connectivity_aware"

    @staticmethod
    def _components(values: np.ndarray, value: int) -> tuple[int, int]:
        mask = values == value
        seen = np.zeros(mask.shape, dtype=bool)
        components = 0
        isolated = 0
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if not mask[row, col] or seen[row, col]:
                    continue
                components += 1
                stack = [(row, col)]
                seen[row, col] = True
                size = 0
                while stack:
                    r, c = stack.pop()
                    size += 1
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if (
                            0 <= nr < mask.shape[0]
                            and 0 <= nc < mask.shape[1]
                            and mask[nr, nc]
                            and not seen[nr, nc]
                        ):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                isolated += int(size == 1)
        return components, isolated

    def _score(self, pattern: int) -> tuple[int, int, int]:
        array = pattern_array(pattern)
        fg_components, fg_isolated = self._components(array, 1)
        bg_components, bg_isolated = self._components(array, 0)
        return fg_components, bg_components, fg_isolated + bg_isolated

    def predict_costs(
        self,
        image: np.ndarray,
        peak: int,
        candidates: list[int],
    ) -> np.ndarray:
        peak_score = self._score(peak)
        return np.asarray(
            [
                sum(abs(a - b) for a, b in zip(self._score(candidate), peak_score))
                + 0.001 * candidate
                for candidate in candidates
            ],
            dtype=np.float64,
        )


def _evaluate_with_tables(
    image: np.ndarray,
    message: list[int],
    tables: dict[int, dict[int, int]],
) -> tuple[int, int, int, float, int, bool, bool]:
    result = embed(
        image,
        message,
        policy="cnn_hamming1",
        mapping_tables=tables,
    )
    wire = serialize_auxiliary(result.auxiliary)
    decoded = deserialize_auxiliary(wire, image_shape=image.shape)
    restored, recovered = extract(result.stego, decoded)
    aux_bits = auxiliary_bits(result.auxiliary)
    return (
        result.embedded_bits,
        aux_bits,
        result.embedded_bits - aux_bits,
        drd(image, result.stego),
        int(np.count_nonzero(image != result.stego)),
        bool(np.array_equal(restored, image)),
        recovered == message,
    )


def run_activation_ablation(
    named_images: list[tuple[str, np.ndarray]],
    ranker: CandidateRanker,
    output_dir: Path,
    payload_fraction: float,
) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for image_index, (name, image) in enumerate(named_images):
        full_tables = build_mapping_tables(
            image,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        )
        variants = [
            ("A0_all_feasible_gross_reported", full_tables, False),
            ("A1_all_feasible_wire_charged", full_tables, True),
            (
                "A2_serialized_net_activation",
                optimize_tables_for_net_capacity(
                    image,
                    full_tables,
                    policy="cnn_hamming1",
                ),
                True,
            ),
        ]
        for variant, tables, charge_wire in variants:
            gross_capacity = capacity_bits(image, tables)
            if gross_capacity <= 0:
                rows.append(
                    {
                        "image": name,
                        "variant": variant,
                        "available": False,
                        "gross_capacity_bits": gross_capacity,
                        "payload_bits": 0,
                        "auxiliary_bits": 0,
                        "net_payload_bits": 0,
                        "posthoc_auxiliary_bits": 0,
                        "posthoc_net_payload_bits": 0,
                        "drd": None,
                        "changed_pixels": 0,
                        "reversible": False,
                        "message_exact": False,
                    }
                )
                continue
            payload = int(gross_capacity * payload_fraction)
            message = rng.integers(
                0,
                2,
                size=payload,
                dtype=np.uint8,
            ).tolist()
            (
                embedded_bits,
                posthoc_aux,
                posthoc_net,
                value_drd,
                changed,
                reversible,
                exact,
            ) = _evaluate_with_tables(image, message, tables)
            rows.append(
                {
                    "image": name,
                    "variant": variant,
                    "available": True,
                    "active_tables": len(tables),
                    "gross_capacity_bits": gross_capacity,
                    "payload_bits": embedded_bits,
                    "auxiliary_bits": posthoc_aux if charge_wire else 0,
                    "net_payload_bits": posthoc_net if charge_wire else embedded_bits,
                    "posthoc_auxiliary_bits": posthoc_aux,
                    "posthoc_net_payload_bits": posthoc_net,
                    "drd": value_drd,
                    "changed_pixels": changed,
                    "reversible": reversible,
                    "message_exact": exact,
                    "seed": SEED + image_index,
                }
            )
    report = {
        "protocol": (
            "Same CNN distance-one candidate assignment; only active-table "
            "selection and whether the exact wire is charged differ."
        ),
        "payload_fraction_of_variant_capacity": payload_fraction,
        "rows": rows,
        "summary": _summary(rows, "variant"),
    }
    _write_json(output_dir / "activation_ablation.json", report)
    _write_csv(output_dir / "activation_ablation.csv", rows)
    return report


def _tables_for_ranker(
    image: np.ndarray,
    method: str,
    ranker: object | None,
) -> dict[int, dict[int, int]]:
    if method == "hamming_index":
        return build_mapping_tables(image, policy="hamming1")
    return build_mapping_tables(
        image,
        policy="cnn_hamming1",
        candidate_ranker=ranker,
    )


def run_ranker_ablation(
    named_images: list[tuple[str, np.ndarray]],
    ranker: CandidateRanker,
    output_dir: Path,
    payload_fraction: float,
) -> dict[str, object]:
    rankers: list[tuple[str, object | None]] = [
        ("hamming_index", None),
        ("direct_drd_cost", DirectDRDRanker()),
        ("transition_preserving", TransitionRanker()),
        ("connectivity_aware", ConnectivityRanker()),
        ("cnn", ranker),
    ]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED + 1)
    for name, image in named_images:
        tables_by_method = {
            method: _tables_for_ranker(image, method, ranker_object)
            for method, ranker_object in rankers
        }
        common_capacity = min(capacity_bits(image, tables) for tables in tables_by_method.values())
        payload = int(common_capacity * payload_fraction)
        message = rng.integers(0, 2, size=payload, dtype=np.uint8).tolist()
        for method, tables in tables_by_method.items():
            try:
                (
                    embedded_bits,
                    aux_bits,
                    net_bits,
                    value_drd,
                    changed,
                    reversible,
                    exact,
                ) = _evaluate_with_tables(image, message, tables)
                rows.append(
                    {
                        "image": name,
                        "ranker": method,
                        "available": True,
                        "common_payload_bits": payload,
                        "gross_capacity_bits": capacity_bits(image, tables),
                        "payload_bits": embedded_bits,
                        "auxiliary_bits": aux_bits,
                        "net_payload_bits": net_bits,
                        "posthoc_auxiliary_bits": aux_bits,
                        "posthoc_net_payload_bits": net_bits,
                        "drd": value_drd,
                        "changed_pixels": changed,
                        "reversible": reversible,
                        "message_exact": exact,
                    }
                )
            except ValueError as error:
                rows.append(
                    {
                        "image": name,
                        "ranker": method,
                        "available": False,
                        "error": str(error),
                    }
                )
    report = {
        "protocol": (
            "All rankers use Hamming-distance-one admissible ZERO patterns and "
            "the same common payload per image; only the candidate ordering differs."
        ),
        "payload_fraction_of_common_capacity": payload_fraction,
        "rows": rows,
        "summary": _summary(rows, "ranker"),
    }
    _write_json(output_dir / "ranker_ablation.json", report)
    _write_csv(output_dir / "ranker_ablation.csv", rows)
    return report


def _context_patch(image: np.ndarray, row: int, col: int, patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    center_row = row + 1
    center_col = col + 1
    row_start = max(0, center_row - radius)
    row_end = min(image.shape[0], center_row + radius + 1)
    col_start = max(0, center_col - radius)
    col_end = min(image.shape[1], center_col + radius + 1)
    patch = image[row_start:row_end, col_start:col_end]
    padding = (
        (max(0, radius - center_row), max(0, center_row + radius + 1 - image.shape[0])),
        (max(0, radius - center_col), max(0, center_col + radius + 1 - image.shape[1])),
    )
    return np.pad(patch, padding, mode="edge") if any(
        value for pair in padding for value in pair
    ) else patch


def _candidate_tensor(
    peak: int,
    candidate: int,
    context: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    center = patch_size // 2
    start = center - 1
    stop = center + 2
    peak_channel = np.zeros((patch_size, patch_size), dtype=np.float32)
    candidate_channel = np.zeros((patch_size, patch_size), dtype=np.float32)
    peak_channel[start:stop, start:stop] = pattern_array(peak)
    candidate_channel[start:stop, start:stop] = pattern_array(candidate)
    return np.stack(
        [peak_channel, candidate_channel, context.astype(np.float32)]
    ).astype(np.float32)


def _ranker_dataset_patch(
    named_images: list[tuple[str, np.ndarray]],
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = []
    targets = []
    groups = []
    for group, (_, image) in enumerate(named_images):
        positions = _pattern_positions(image)
        zeros = [pattern for pattern in range(512) if pattern not in positions]
        costs = flip_cost_map(image)
        peaks = [
            pattern
            for pattern, occurrences in positions.items()
            if len(occurrences) >= 2 and pattern not in (0, 511)
        ]
        peaks.sort(key=lambda pattern: (-len(positions[pattern]), pattern))
        for peak in peaks:
            candidates = sorted(
                (
                    candidate
                    for candidate in zeros
                    if hamming(peak, candidate) <= 2
                ),
                key=lambda candidate: (hamming(peak, candidate), candidate),
            )[:16]
            if len(candidates) < 2:
                continue
            context = np.mean(
                [
                    _context_patch(image, row, col, patch_size)
                    for row, col in positions[peak]
                ],
                axis=0,
                dtype=np.float32,
            )
            for candidate in candidates:
                tensors.append(_candidate_tensor(peak, candidate, context, patch_size))
                targets.append(
                    empirical_replacement_cost(
                        image,
                        positions[peak],
                        peak,
                        candidate,
                        costs,
                    )
                )
                groups.append(group)
    return (
        np.asarray(tensors, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(groups, dtype=np.int32),
    )


class PatchCNN(nn.Module):
    def __init__(self, patch_size: int, layers: int, width: int = 24) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        in_channels = 3
        for index in range(layers):
            out_channels = width * (2 if index == layers - 1 and layers > 1 else 1)
            modules.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        modules.append(nn.AdaptiveAvgPool2d((3, 3)))
        self.features = nn.Sequential(*modules)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(values)).squeeze(1)


@dataclass
class PatchRanker:
    model: PatchCNN
    patch_size: int
    _cache: dict[
        tuple[tuple[int, int], bytes],
        tuple[dict[int, list[tuple[int, int]]], dict[int, np.ndarray]],
    ] = field(default_factory=dict, init=False, repr=False)

    def _image_cache(
        self,
        image: np.ndarray,
    ) -> tuple[dict[int, list[tuple[int, int]]], dict[int, np.ndarray]]:
        binary = np.ascontiguousarray(image.astype(np.uint8, copy=False))
        key = (binary.shape, binary.tobytes())
        if key not in self._cache:
            positions = _pattern_positions(binary)
            contexts = {
                peak: np.mean(
                    [
                        _context_patch(binary, row, col, self.patch_size)
                        for row, col in peak_positions
                    ],
                    axis=0,
                    dtype=np.float32,
                )
                for peak, peak_positions in positions.items()
                if peak_positions
            }
            self._cache[key] = (positions, contexts)
        return self._cache[key]

    def predict_costs(
        self,
        image: np.ndarray,
        peak: int,
        candidates: list[int],
    ) -> np.ndarray:
        positions, contexts = self._image_cache(image)
        peak_positions = positions.get(peak, [])
        if not peak_positions:
            raise ValueError("Peak does not occur in image")
        context = contexts[peak]
        tensors = np.stack(
            [
                _candidate_tensor(peak, candidate, context, self.patch_size)
                for candidate in candidates
            ]
        )
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(tensors)).numpy()


def _train_patch_model(
    tensors: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    layers: int,
    patch_size: int,
    width: int,
    epochs: int,
    seed: int,
) -> tuple[PatchCNN, float, int]:
    torch.manual_seed(seed)
    model = PatchCNN(patch_size=patch_size, layers=layers, width=width)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(tensors[train_mask]),
            torch.from_numpy(targets[train_mask]),
        ),
        batch_size=128,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_function = nn.SmoothL1Loss()
    started = perf_counter()
    for _ in range(epochs):
        model.train()
        for features, target in loader:
            prediction = model(features)
            loss = loss_function(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval(), perf_counter() - started, sum(
        parameter.numel() for parameter in model.parameters()
    )


def run_architecture_ablation(
    named_images: list[tuple[str, np.ndarray]],
    output_dir: Path,
    epochs: int,
) -> dict[str, object]:
    protocol = (
        "Diagnostic validation on the same two held-out images used by the "
        "two-image detailed split. All configurations use AdamW, learning "
        "rate 1e-3, SmoothL1 loss, batch size 128, identical train/validation "
        "groups, deterministic seeds, and the requested epoch count; LOSO "
        "placement remains the main CNN result."
    )
    configs = [
        ("shallow_1x9_w24", 9, 1, 24),
        ("current_2x9_w24", 9, 2, 24),
        ("deep_3x9_w24", 9, 3, 24),
        ("patch_7_2x_w24", 7, 2, 24),
        ("patch_11_2x_w24", 11, 2, 24),
        ("narrow_2x9_w16", 9, 2, 16),
        ("wide_2x9_w32", 9, 2, 32),
    ]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED + 2)
    for name, patch_size, layers, width in configs:
        tensors, targets, groups = _ranker_dataset_patch(named_images, patch_size)
        validation_mask = np.isin(groups, (6, 7))
        train_mask = ~validation_mask
        model, train_seconds, params = _train_patch_model(
            tensors,
            targets,
            train_mask,
            layers,
            patch_size,
            width,
            epochs,
            SEED + layers + patch_size + width,
        )
        inference_started = perf_counter()
        with torch.no_grad():
            predictions = model(torch.from_numpy(tensors[validation_mask])).numpy()
        inference_seconds = perf_counter() - inference_started
        validation_targets = targets[validation_mask]
        pearson = float(pearsonr(validation_targets, predictions).statistic)
        spearman = float(spearmanr(validation_targets, predictions).statistic)

        placement_rows = []
        ranker = PatchRanker(model, patch_size)
        heldout = [named_images[6], named_images[7]]
        for image_name, image in heldout:
            hamming_tables = build_mapping_tables(image, policy="hamming1")
            model_tables = build_mapping_tables(
                image,
                policy="cnn_hamming1",
                candidate_ranker=ranker,
            )
            payload = int(
                min(capacity_bits(image, hamming_tables), capacity_bits(image, model_tables))
                * 0.75
            )
            message = rng.integers(0, 2, size=payload, dtype=np.uint8).tolist()
            _, _, _, hamming_drd, _, _, _ = _evaluate_with_tables(
                image,
                message,
                hamming_tables,
            )
            _, _, _, model_drd, _, reversible, exact = _evaluate_with_tables(
                image,
                message,
                model_tables,
            )
            placement_rows.append(
                {
                    "image": image_name,
                    "hamming_drd": hamming_drd,
                    "model_drd": model_drd,
                    "reversible": reversible,
                    "message_exact": exact,
                }
            )
        rows.append(
            {
                "configuration": name,
                "patch_size": patch_size,
                "conv_layers": layers,
                "base_width": width,
                "epochs": epochs,
                "parameters": params,
                "training_seconds": train_seconds,
                "validation_inference_seconds": inference_seconds,
                "inference_microseconds_per_candidate": (
                    1_000_000.0 * inference_seconds / int(validation_mask.sum())
                ),
                "validation_samples": int(validation_mask.sum()),
                "mae": float(np.mean(np.abs(validation_targets - predictions))),
                "pearson": pearson,
                "spearman": spearman,
                "heldout_mean_hamming_drd": float(
                    np.mean([row["hamming_drd"] for row in placement_rows])
                ),
                "heldout_mean_model_drd": float(
                    np.mean([row["model_drd"] for row in placement_rows])
                ),
                "all_reversible": all(bool(row["reversible"]) for row in placement_rows),
                "all_messages_exact": all(
                    bool(row["message_exact"]) for row in placement_rows
                ),
            }
        )
        partial_report = {
            "protocol": protocol,
            "complete": len(rows) == len(configs),
            "rows": rows,
        }
        _write_json(output_dir / "cnn_architecture_ablation.json", partial_report)
        _write_csv(output_dir / "cnn_architecture_ablation.csv", rows)
    report = {
        "protocol": protocol,
        "complete": True,
        "rows": rows,
    }
    _write_json(output_dir / "cnn_architecture_ablation.json", report)
    _write_csv(output_dir / "cnn_architecture_ablation.csv", rows)
    return report


def run_module_runtime(
    named_images: list[tuple[str, np.ndarray]],
    ranker: CandidateRanker,
    output_dir: Path,
) -> dict[str, object]:
    rng = np.random.default_rng(SEED + 3)
    rows: list[dict[str, object]] = []
    for name, image in named_images:
        started = perf_counter()
        full_tables = build_mapping_tables(
            image,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        )
        after_candidates = perf_counter()
        tables = optimize_tables_for_net_capacity(
            image,
            full_tables,
            policy="cnn_hamming1",
        )
        after_selection = perf_counter()
        payload = int(capacity_bits(image, tables) * 0.75)
        message = rng.integers(0, 2, size=payload, dtype=np.uint8).tolist()
        result = embed(
            image,
            message,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
            mapping_tables=tables,
        )
        after_embedding = perf_counter()
        wire = serialize_auxiliary(result.auxiliary)
        after_serialization = perf_counter()
        decoded = deserialize_auxiliary(wire, image_shape=image.shape)
        restored, recovered = extract(result.stego, decoded)
        after_extraction = perf_counter()
        reversible = bool(np.array_equal(restored, image))
        exact = recovered == message
        after_verification = perf_counter()
        rows.append(
            {
                "image": name,
                "candidate_generation_s": after_candidates - started,
                "prefix_optimization_s": after_selection - after_candidates,
                "embedding_s": after_embedding - after_selection,
                "serialization_s": after_serialization - after_embedding,
                "extraction_s": after_extraction - after_serialization,
                "verification_s": after_verification - after_extraction,
                "total_s": after_verification - started,
                "payload_bits": payload,
                "auxiliary_bits": auxiliary_bits(result.auxiliary),
                "reversible": reversible,
                "message_exact": exact,
            }
        )
    summary = {}
    for key in rows[0]:
        if key.endswith("_s") or key in ("payload_bits", "auxiliary_bits"):
            summary[f"median_{key}"] = float(np.median([float(row[key]) for row in rows]))
            summary[f"mean_{key}"] = float(np.mean([float(row[key]) for row in rows]))
    report = {"rows": rows, "summary": summary}
    _write_json(output_dir / "module_runtime.json", report)
    _write_csv(output_dir / "module_runtime.csv", rows)
    return report


def _pattern_distribution(image: np.ndarray) -> np.ndarray:
    histogram = np.zeros(512, dtype=np.float64)
    for _, _, block in iter_blocks(image):
        histogram[pattern_index(block)] += 1
    total = histogram.sum()
    return histogram / total if total else histogram


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    eps = 1e-12
    p = left + eps
    q = right + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def run_payload_diagnostics(
    named_images: list[tuple[str, np.ndarray]],
    ranker: CandidateRanker,
    output_dir: Path,
) -> dict[str, object]:
    rng = np.random.default_rng(SEED + 4)
    fractions = (0.25, 0.5, 0.75, 1.0)
    rows: list[dict[str, object]] = []
    for name, image in named_images:
        cover_distribution = _pattern_distribution(image)
        cover_horizontal = float(np.mean(image[:, 1:] != image[:, :-1]))
        cover_vertical = float(np.mean(image[1:, :] != image[:-1, :]))
        tables = optimize_tables_for_net_capacity(
            image,
            build_mapping_tables(
                image,
                policy="cnn_hamming1",
                candidate_ranker=ranker,
            ),
            policy="cnn_hamming1",
        )
        maximum = capacity_bits(image, tables)
        for fraction in fractions:
            payload = int(maximum * fraction)
            message = rng.integers(0, 2, size=payload, dtype=np.uint8).tolist()
            result = embed(
                image,
                message,
                policy="cnn_hamming1",
                candidate_ranker=ranker,
                mapping_tables=tables,
            )
            stego_distribution = _pattern_distribution(result.stego)
            stego_horizontal = float(np.mean(result.stego[:, 1:] != result.stego[:, :-1]))
            stego_vertical = float(np.mean(result.stego[1:, :] != result.stego[:-1, :]))
            rows.append(
                {
                    "image": name,
                    "payload_fraction": fraction,
                    "payload_bits": payload,
                    "auxiliary_bits": auxiliary_bits(result.auxiliary),
                    "net_payload_bits": payload - auxiliary_bits(result.auxiliary),
                    "drd": drd(image, result.stego),
                    "pattern_js_divergence": _js_divergence(
                        cover_distribution,
                        stego_distribution,
                    ),
                    "delta_horizontal_transitions": stego_horizontal - cover_horizontal,
                    "delta_vertical_transitions": stego_vertical - cover_vertical,
                    "changed_pixels": int(np.count_nonzero(image != result.stego)),
                }
            )
    report = {
        "protocol": (
            "Document-image payload sweep for explaining statistical footprint; "
            "AUC values remain those reported in the grouped BOSSbase protocol."
        ),
        "rows": rows,
    }
    _write_json(output_dir / "payload_diagnostics.json", report)
    _write_csv(output_dir / "payload_diagnostics.csv", rows)
    return report


def run_visual_comparison(
    named_images: list[tuple[str, np.ndarray]],
    ranker: CandidateRanker,
    output_dir: Path,
    figure_dir: Path,
) -> dict[str, object]:
    name, image = next(item for item in named_images if item[0] == "table1-3")
    payload = 512
    rng = np.random.default_rng(SEED + 5)
    message = rng.integers(0, 2, size=payload, dtype=np.uint8).tolist()
    rows = []
    stegos: dict[str, np.ndarray] = {}

    tables = build_mapping_tables(
        image,
        policy="cnn_hamming1",
        candidate_ranker=ranker,
    )
    tables = optimize_tables_for_net_capacity(image, tables, policy="cnn_hamming1")
    abm_result = embed(
        image,
        message,
        policy="cnn_hamming1",
        candidate_ranker=ranker,
        mapping_tables=tables,
    )
    stegos["ABM"] = abm_result.stego
    rows.append({"method": "ABM", "drd": drd(image, abm_result.stego)})

    profile = fit_ppocp_profile(
        image_item for image_name, image_item in named_images if image_name != name
    )
    for method in ("Dong", "Huynh-Nguyen", "PPOCP"):
        try:
            if method == "Dong":
                candidates = []
                for divisor in range(1, 11):
                    try:
                        result = embed_dong(image, message, context_divisor=divisor)
                        candidates.append((drd(image, result.stego), divisor, result.stego))
                    except ValueError:
                        continue
                if not candidates:
                    raise ValueError("capacity below target")
                _, divisor, stego = min(candidates, key=lambda item: (item[0], item[1]))
                parameter = f"l={divisor}"
            elif method == "Huynh-Nguyen":
                if huynh_capacity_bits(image) < payload:
                    raise ValueError("capacity below target")
                stego = embed_huynh(image, message).stego
                parameter = "T=5"
            else:
                stego = embed_ppocp(image, message, profile).stego
                parameter = ""
            stegos[method] = stego
            rows.append({"method": method, "drd": drd(image, stego), "parameter": parameter})
        except ValueError as error:
            rows.append({"method": method, "available": False, "error": str(error)})

    available = [(method, stego) for method, stego in stegos.items()]
    union = np.zeros_like(image, dtype=np.uint8)
    for _, stego in available:
        union |= (image != stego).astype(np.uint8)
    crop_height, crop_width = 120, 140
    if int(union.sum()) > 0:
        integral = (
            np.pad(union.astype(np.int64), ((1, 0), (1, 0)))
            .cumsum(axis=0)
            .cumsum(axis=1)
        )
        best_score = -1
        best_row = 0
        best_col = 0
        for row in range(0, image.shape[0] - crop_height + 1, 6):
            for col in range(0, image.shape[1] - crop_width + 1, 6):
                score = (
                    integral[row + crop_height, col + crop_width]
                    - integral[row, col + crop_width]
                    - integral[row + crop_height, col]
                    + integral[row, col]
                )
                if int(score) > best_score:
                    best_score = int(score)
                    best_row = row
                    best_col = col
    else:
        best_row, best_col = 110, 120
    crop = (slice(best_row, best_row + crop_height), slice(best_col, best_col + crop_width))
    fig, axes = plt.subplots(len(available), 3, figsize=(6.8, 2.1 * len(available)))
    if len(available) == 1:
        axes = np.asarray([axes])
    for row_index, (method, stego) in enumerate(available):
        cover_crop = image[crop]
        stego_crop = stego[crop]
        residual = cover_crop != stego_crop
        axes[row_index, 0].imshow(cover_crop, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].imshow(stego_crop, cmap="gray", vmin=0, vmax=1)
        residual_rgb = np.ones((*residual.shape, 3), dtype=np.float32)
        residual_rgb[residual] = np.array([0.85, 0.0, 0.0], dtype=np.float32)
        axes[row_index, 2].imshow(residual_rgb)
        axes[row_index, 0].set_ylabel(method, fontsize=8)
        for col, title in enumerate(("Cover", "Marked", "Residual")):
            axes[row_index, col].set_title(title, fontsize=8)
            axes[row_index, col].set_xticks([])
            axes[row_index, col].set_yticks([])
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "Figure_10.pdf")
    fig.savefig(figure_dir / "Figure_10.png", dpi=220)
    plt.close(fig)
    report = {
        "image": name,
        "payload_bits": payload,
        "crop": f"rows {best_row}:{best_row + crop_height}, cols {best_col}:{best_col + crop_width}",
        "rows": rows,
        "figure_pdf": str(figure_dir / "Figure_10.pdf"),
    }
    _write_json(output_dir / "visual_comparison.json", report)
    return report


def _save_binary_image(path: Path, image: np.ndarray, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image.astype(np.uint8) * 255), mode="L").save(path, **kwargs)


def _load_binary_image(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return (np.asarray(handle.convert("L")) >= 128).astype(np.uint8)


def run_channel_stress(
    named_images: list[tuple[str, np.ndarray]],
    ranker: CandidateRanker,
    output_dir: Path,
) -> dict[str, object]:
    """Verify lossless containers and document expected lossy/noisy failure."""

    name, image = next(item for item in named_images if item[0] == "table1-3")
    tables = optimize_tables_for_net_capacity(
        image,
        build_mapping_tables(
            image,
            policy="cnn_hamming1",
            candidate_ranker=ranker,
        ),
        policy="cnn_hamming1",
    )
    payload = min(256, capacity_bits(image, tables))
    message = (
        np.random.default_rng(SEED)
        .integers(0, 2, size=payload, dtype=np.uint8)
        .tolist()
    )
    result = embed(
        image,
        message,
        policy="cnn_hamming1",
        candidate_ranker=ranker,
        mapping_tables=tables,
    )
    wire = serialize_auxiliary(result.auxiliary)
    aux = deserialize_auxiliary(wire, image_shape=image.shape)

    stress_dir = output_dir / "channel_stress"
    png_path = stress_dir / f"{name}_abm.png"
    tiff_path = stress_dir / f"{name}_abm.tiff"
    jpeg_path = stress_dir / f"{name}_abm_q05.jpg"
    _save_binary_image(png_path, result.stego)
    _save_binary_image(tiff_path, result.stego)
    _save_binary_image(jpeg_path, quality=5, image=result.stego)

    def round_trip(candidate: np.ndarray) -> tuple[bool, bool, str]:
        try:
            restored, recovered = extract(candidate, aux)
            return (
                bool(np.array_equal(restored, image)),
                recovered == message,
                "",
            )
        except Exception as error:  # noqa: BLE001 - status is part of the stress test.
            return False, False, str(error)

    png_recovery, png_message, png_error = round_trip(_load_binary_image(png_path))
    tiff_recovery, tiff_message, tiff_error = round_trip(_load_binary_image(tiff_path))
    jpeg_binary = _load_binary_image(jpeg_path)
    jpeg_recovery, jpeg_message, jpeg_error = round_trip(jpeg_binary)

    corrupted = result.stego.copy()
    inverse_patterns = set()
    for peak, table in result.auxiliary.mapping_tables.items():
        inverse_patterns.add(peak)
        inverse_patterns.update(table.values())
    corrupted_location = None
    for row, col, block in iter_blocks(result.stego):
        current = pattern_index(block)
        if current not in inverse_patterns:
            continue
        for local in range(9):
            trial = block.copy().reshape(-1)
            trial[local] ^= 1
            if pattern_index(trial.reshape(3, 3)) not in inverse_patterns:
                rr = row + local // 3
                cc = col + local % 3
                corrupted[rr, cc] ^= 1
                corrupted_location = [int(rr), int(cc)]
                break
        if corrupted_location is not None:
            break
    noise_recovery, noise_message, noise_error = round_trip(corrupted)

    report = {
        "protocol": (
            "ABM is tested with its auxiliary stream unchanged after container "
            "round trips or channel perturbations."
        ),
        "image": name,
        "payload_bits": payload,
        "auxiliary_bits": auxiliary_bits(result.auxiliary),
        "png": {
            "binary_stego_preserved": bool(np.array_equal(_load_binary_image(png_path), result.stego)),
            "cover_recovered_exactly": png_recovery,
            "message_recovered_exactly": png_message,
            "error": png_error,
        },
        "tiff": {
            "binary_stego_preserved": bool(np.array_equal(_load_binary_image(tiff_path), result.stego)),
            "cover_recovered_exactly": tiff_recovery,
            "message_recovered_exactly": tiff_message,
            "error": tiff_error,
        },
        "jpeg_q05": {
            "binary_stego_preserved": bool(np.array_equal(jpeg_binary, result.stego)),
            "cover_recovered_exactly": jpeg_recovery,
            "message_recovered_exactly": jpeg_message,
            "error": jpeg_error,
        },
        "single_pixel_noise": {
            "corrupted_location": corrupted_location,
            "binary_stego_preserved": bool(np.array_equal(corrupted, result.stego)),
            "cover_recovered_exactly": noise_recovery,
            "message_recovered_exactly": noise_message,
            "error": noise_error,
        },
    }
    _write_json(output_dir / "channel_stress.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=DOCUMENT_IMAGE_DIR)
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=Path("ml_ranker_results/candidate_ranker.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision_results"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("paper/images"),
    )
    parser.add_argument("--payload-fraction", type=float, default=0.75)
    parser.add_argument("--architecture-epochs", type=int, default=20)
    parser.add_argument(
        "--only",
        choices=("all", "architecture"),
        default="all",
        help="Generate all revision artifacts or only the CNN architecture table.",
    )
    args = parser.parse_args()

    named_images = [
        (Path(filename).stem, load_binary(args.image_dir / filename))
        for filename in DEFAULT_IMAGES
    ]
    ranker = CandidateRanker.load(args.ranker_model)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.only == "architecture":
        report = run_architecture_ablation(
            named_images,
            output_dir,
            args.architecture_epochs,
        )
        print(json.dumps(report, indent=2))
        return

    reports = {
        "activation_ablation": run_activation_ablation(
            named_images,
            ranker,
            output_dir,
            args.payload_fraction,
        ),
        "ranker_ablation": run_ranker_ablation(
            named_images,
            ranker,
            output_dir,
            args.payload_fraction,
        ),
        "cnn_architecture_ablation": run_architecture_ablation(
            named_images,
            output_dir,
            args.architecture_epochs,
        ),
        "module_runtime": run_module_runtime(named_images, ranker, output_dir),
        "payload_diagnostics": run_payload_diagnostics(
            named_images,
            ranker,
            output_dir,
        ),
        "visual_comparison": run_visual_comparison(
            named_images,
            ranker,
            output_dir,
            args.figure_dir,
        ),
        "channel_stress": run_channel_stress(
            named_images,
            ranker,
            output_dir,
        ),
    }
    _write_json(output_dir / "revision_ablation_summary.json", reports)
    compact = {
        key: value.get("summary", value.get("rows", value))
        for key, value in reports.items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

