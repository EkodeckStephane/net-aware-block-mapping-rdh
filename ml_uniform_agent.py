"""Random-Forest agent for context-aware uniform-block embedding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import convolve2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict

from abm_rdh import BLOCK_SIZE, DRD_WEIGHTS, as_binary


SAFE_COST_THRESHOLD = 0.5
CONTEXT_RADIUS = 4


@dataclass(frozen=True)
class UniformBlockCandidate:
    row: int
    col: int
    flip_position: int
    safe_probability: float


@dataclass(frozen=True)
class UniformAgentMetrics:
    roc_auc: float
    precision: float
    recall: float
    position_accuracy: float
    samples: int
    safe_samples: int


@dataclass
class UniformBlockAgent:
    safe_model: RandomForestClassifier
    position_model: RandomForestClassifier
    safe_cost_threshold: float = SAFE_COST_THRESHOLD

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "UniformBlockAgent":
        return joblib.load(path)


def _context_patch(image: np.ndarray, row: int, col: int) -> np.ndarray:
    center_row = row + 1
    center_col = col + 1
    row_start = max(0, center_row - CONTEXT_RADIUS)
    row_end = min(image.shape[0], center_row + CONTEXT_RADIUS + 1)
    col_start = max(0, center_col - CONTEXT_RADIUS)
    col_end = min(image.shape[1], center_col + CONTEXT_RADIUS + 1)
    patch = image[row_start:row_end, col_start:col_end]
    padding = (
        (max(0, CONTEXT_RADIUS - center_row), max(0, center_row + CONTEXT_RADIUS + 1 - image.shape[0])),
        (max(0, CONTEXT_RADIUS - center_col), max(0, center_col + CONTEXT_RADIUS + 1 - image.shape[1])),
    )
    if any(value for pair in padding for value in pair):
        patch = np.pad(patch, padding, mode="edge")
    return patch


def context_features(image: np.ndarray, row: int, col: int) -> np.ndarray:
    patch = _context_patch(image, row, col).astype(np.float32)
    block_value = float(image[row, col])
    horizontal_changes = np.count_nonzero(patch[:, 1:] != patch[:, :-1])
    vertical_changes = np.count_nonzero(patch[1:, :] != patch[:-1, :])
    density = float(patch.mean())
    summary = np.array(
        [block_value, density, horizontal_changes / 72.0, vertical_changes / 72.0],
        dtype=np.float32,
    )
    return np.concatenate([patch.reshape(-1), summary])


def invariant_uniform_features(
    image: np.ndarray, row: int, col: int
) -> tuple[np.ndarray, int] | None:
    """Return features invariant to at most one flip in the central 3x3 block."""

    patch = _context_patch(image, row, col).astype(np.float32)
    block = patch[3:6, 3:6]
    ones = int(block.sum())
    if ones not in (0, 1, 8, 9):
        return None
    original_value = int(ones >= 8)
    patch[3:6, 3:6] = original_value
    horizontal_changes = np.count_nonzero(patch[:, 1:] != patch[:, :-1])
    vertical_changes = np.count_nonzero(patch[1:, :] != patch[:-1, :])
    summary = np.array(
        [
            float(original_value),
            float(patch.mean()),
            horizontal_changes / 72.0,
            vertical_changes / 72.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([patch.reshape(-1), summary]), original_value


def local_flip_cost(image: np.ndarray, row: int, col: int) -> float:
    flipped_value = 1 - int(image[row, col])
    height, width = image.shape
    cost = 0.0
    for delta_row in range(-2, 3):
        for delta_col in range(-2, 3):
            source_row = row + delta_row
            source_col = col + delta_col
            if 0 <= source_row < height and 0 <= source_col < width:
                cost += (
                    abs(int(image[source_row, source_col]) - flipped_value)
                    * DRD_WEIGHTS[delta_row + 2, delta_col + 2]
                )
    return float(cost)


def best_uniform_flip(image: np.ndarray, row: int, col: int) -> tuple[int, float]:
    costs = [
        local_flip_cost(image, row + index // 3, col + index % 3)
        for index in range(9)
    ]
    best_position = int(np.argmin(costs))
    return best_position, float(costs[best_position])


def flip_cost_map(image: np.ndarray) -> np.ndarray:
    binary = as_binary(image).astype(np.float64)
    weighted_ones = convolve2d(
        binary,
        DRD_WEIGHTS,
        mode="same",
        boundary="fill",
        fillvalue=0,
    )
    available_weight = convolve2d(
        np.ones_like(binary),
        DRD_WEIGHTS,
        mode="same",
        boundary="fill",
        fillvalue=0,
    )
    return np.where(binary == 0, available_weight - weighted_ones, weighted_ones)


def uniform_training_samples(
    named_images: list[tuple[str, np.ndarray]],
    *,
    safe_cost_threshold: float = SAFE_COST_THRESHOLD,
    unsafe_ratio: int = 5,
    seed: int = 20260607,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = []
    labels = []
    positions = []
    groups = []

    for group_index, (_, raw_image) in enumerate(named_images):
        image = as_binary(raw_image)
        costs = flip_cost_map(image)
        safe_rows = []
        unsafe_rows = []
        redundant_uniform_rows = []
        for row in range(0, image.shape[0] - 2, BLOCK_SIZE):
            for col in range(0, image.shape[1] - 2, BLOCK_SIZE):
                block = image[row : row + 3, col : col + 3]
                if block.min() != block.max():
                    continue
                feature = context_features(image, row, col)
                if np.all(feature[:81] == feature[0]):
                    if len(redundant_uniform_rows) < 1000:
                        redundant_uniform_rows.append((feature, 0, 0, group_index))
                    continue
                block_costs = costs[row : row + 3, col : col + 3].reshape(-1)
                position = int(np.argmin(block_costs))
                cost = float(block_costs[position])
                sample = (
                    feature,
                    int(cost <= safe_cost_threshold),
                    position,
                    group_index,
                )
                (safe_rows if sample[1] else unsafe_rows).append(sample)

        if safe_rows:
            unsafe_count = min(len(unsafe_rows), unsafe_ratio * len(safe_rows))
            selected_indices = rng.choice(
                len(unsafe_rows), size=unsafe_count, replace=False
            )
            selected = (
                safe_rows
                + [unsafe_rows[index] for index in selected_indices]
                + redundant_uniform_rows[: min(len(redundant_uniform_rows), len(safe_rows))]
            )
        else:
            selected = unsafe_rows[:100]
        rng.shuffle(selected)
        for feature, label, position, group in selected:
            features.append(feature)
            labels.append(label)
            positions.append(position)
            groups.append(group)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.uint8),
        np.asarray(positions, dtype=np.uint8),
        np.asarray(groups, dtype=np.int32),
    )


def train_uniform_agent(
    named_images: list[tuple[str, np.ndarray]],
    *,
    safe_cost_threshold: float = SAFE_COST_THRESHOLD,
    seed: int = 20260607,
) -> tuple[UniformBlockAgent, UniformAgentMetrics]:
    features, labels, positions, groups = uniform_training_samples(
        named_images,
        safe_cost_threshold=safe_cost_threshold,
        seed=seed,
    )
    folds = min(3, len(np.unique(groups)))
    cross_validation = GroupKFold(n_splits=folds)

    safe_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    safe_probabilities = cross_val_predict(
        safe_model,
        features,
        labels,
        groups=groups,
        cv=cross_validation,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    safe_predictions = safe_probabilities >= 0.5

    safe_mask = labels == 1
    position_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    safe_groups = groups[safe_mask]
    position_cv = GroupKFold(n_splits=min(folds, len(np.unique(safe_groups))))
    position_predictions = cross_val_predict(
        position_model,
        features[safe_mask],
        positions[safe_mask],
        groups=safe_groups,
        cv=position_cv,
        n_jobs=-1,
    )

    metrics = UniformAgentMetrics(
        roc_auc=float(roc_auc_score(labels, safe_probabilities)),
        precision=float(precision_score(labels, safe_predictions, zero_division=0)),
        recall=float(recall_score(labels, safe_predictions, zero_division=0)),
        position_accuracy=float(
            accuracy_score(positions[safe_mask], position_predictions)
        ),
        samples=int(labels.size),
        safe_samples=int(safe_mask.sum()),
    )
    safe_model.fit(features, labels)
    position_model.fit(features[safe_mask], positions[safe_mask])
    return (
        UniformBlockAgent(
            safe_model=safe_model,
            position_model=position_model,
            safe_cost_threshold=safe_cost_threshold,
        ),
        metrics,
    )


def find_safe_uniform_blocks(
    image: np.ndarray,
    agent: UniformBlockAgent,
    *,
    probability_threshold: float = 0.7,
) -> list[UniformBlockCandidate]:
    binary = as_binary(image)
    locations = []
    feature_rows = []
    for row in range(0, binary.shape[0] - 2, BLOCK_SIZE):
        for col in range(0, binary.shape[1] - 2, BLOCK_SIZE):
            block = binary[row : row + 3, col : col + 3]
            if block.min() != block.max():
                continue
            locations.append((row, col))
            feature_rows.append(context_features(binary, row, col))

    if not feature_rows:
        return []
    feature_matrix = np.asarray(feature_rows, dtype=np.float32)
    probabilities = agent.safe_model.predict_proba(feature_matrix)[:, 1]
    selected_indices = np.flatnonzero(probabilities >= probability_threshold)
    if selected_indices.size == 0:
        return []
    positions = agent.position_model.predict(feature_matrix[selected_indices])
    candidates = [
        UniformBlockCandidate(
            locations[index][0],
            locations[index][1],
            int(position),
            float(probabilities[index]),
        )
        for index, position in zip(selected_indices, positions)
    ]
    return candidates
