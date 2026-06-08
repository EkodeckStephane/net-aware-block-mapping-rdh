"""CNN ranker for perceptual PEAK-to-ZERO replacement cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from abm_rdh import (
    BLOCK_SIZE,
    PATTERN_COUNT,
    as_binary,
    hamming,
    pattern_array,
    pattern_index,
)
from ml_uniform_agent import flip_cost_map


@dataclass(frozen=True)
class RankerMetrics:
    validation_mae: float
    validation_pearson: float
    training_samples: int
    validation_samples: int


class CandidateCostCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(values)).squeeze(1)


@dataclass
class CandidateRanker:
    model: CandidateCostCNN
    _positions_cache: dict[int, dict[int, list[tuple[int, int]]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _image_refs: dict[int, np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _context_cache: dict[tuple[int, int], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict()}, path)

    @staticmethod
    def load(path: Path) -> "CandidateRanker":
        model = CandidateCostCNN()
        payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return CandidateRanker(model)

    def predict_costs(
        self,
        image: np.ndarray,
        peak: int,
        candidates: list[int],
    ) -> np.ndarray:
        image_key = id(image)
        if image_key not in self._positions_cache:
            self._image_refs[image_key] = image
            self._positions_cache[image_key] = _pattern_positions(image)
        context_key = (image_key, peak)
        if context_key not in self._context_cache:
            positions = self._positions_cache[image_key].get(peak, [])
            if not positions:
                raise ValueError("Peak does not occur in image")
            self._context_cache[context_key] = np.mean(
                [_context9(image, row, col) for row, col in positions],
                axis=0,
                dtype=np.float32,
            )
        context = self._context_cache[context_key]
        tensors = np.stack(
            [candidate_tensor(peak, candidate, context) for candidate in candidates]
        )
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(tensors)).numpy()


def _pattern_positions(image: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    positions: dict[int, list[tuple[int, int]]] = {}
    for row in range(0, image.shape[0] - 2, BLOCK_SIZE):
        for col in range(0, image.shape[1] - 2, BLOCK_SIZE):
            pattern = pattern_index(image[row : row + 3, col : col + 3])
            positions.setdefault(pattern, []).append((row, col))
    return positions


def _context9(image: np.ndarray, row: int, col: int) -> np.ndarray:
    center_row = row + 1
    center_col = col + 1
    row_start = max(0, center_row - 4)
    row_end = min(image.shape[0], center_row + 5)
    col_start = max(0, center_col - 4)
    col_end = min(image.shape[1], center_col + 5)
    patch = image[row_start:row_end, col_start:col_end]
    padding = (
        (max(0, 4 - center_row), max(0, center_row + 5 - image.shape[0])),
        (max(0, 4 - center_col), max(0, center_col + 5 - image.shape[1])),
    )
    return np.pad(patch, padding, mode="edge") if any(
        value for pair in padding for value in pair
    ) else patch


def average_peak_context(image: np.ndarray, peak: int) -> np.ndarray:
    binary = as_binary(image)
    positions = _pattern_positions(binary).get(peak, [])
    if not positions:
        raise ValueError("Peak does not occur in image")
    contexts = [_context9(binary, row, col) for row, col in positions]
    return np.mean(contexts, axis=0, dtype=np.float32)


def candidate_tensor(peak: int, candidate: int, context: np.ndarray) -> np.ndarray:
    peak_channel = np.zeros((9, 9), dtype=np.float32)
    candidate_channel = np.zeros((9, 9), dtype=np.float32)
    peak_channel[3:6, 3:6] = pattern_array(peak)
    candidate_channel[3:6, 3:6] = pattern_array(candidate)
    return np.stack(
        [peak_channel, candidate_channel, context.astype(np.float32)]
    ).astype(np.float32)


def empirical_replacement_cost(
    image: np.ndarray,
    positions: list[tuple[int, int]],
    peak: int,
    candidate: int,
    costs: np.ndarray | None = None,
) -> float:
    if costs is None:
        costs = flip_cost_map(image)
    changed = [
        index
        for index, (left, right) in enumerate(
            zip(pattern_array(peak).reshape(-1), pattern_array(candidate).reshape(-1))
        )
        if left != right
    ]
    total = 0.0
    for row, col in positions:
        total += sum(
            float(costs[row + index // 3, col + index % 3]) for index in changed
        )
    return total / max(len(positions), 1)


def ranker_dataset(
    named_images: list[tuple[str, np.ndarray]],
    *,
    max_hamming: int = 2,
    max_candidates_per_peak: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = []
    targets = []
    groups = []
    for group, (_, raw_image) in enumerate(named_images):
        image = as_binary(raw_image)
        positions = _pattern_positions(image)
        zeros = [pattern for pattern in range(PATTERN_COUNT) if pattern not in positions]
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
                    if hamming(peak, candidate) <= max_hamming
                ),
                key=lambda candidate: (hamming(peak, candidate), candidate),
            )[:max_candidates_per_peak]
            if len(candidates) < 2:
                continue
            context = np.mean(
                [_context9(image, row, col) for row, col in positions[peak]],
                axis=0,
                dtype=np.float32,
            )
            for candidate in candidates:
                tensors.append(candidate_tensor(peak, candidate, context))
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


def train_candidate_ranker(
    named_images: list[tuple[str, np.ndarray]],
    *,
    validation_groups: tuple[int, ...] = (6, 7),
    epochs: int = 20,
    seed: int = 20260607,
) -> tuple[CandidateRanker, CandidateRanker, RankerMetrics]:
    torch.manual_seed(seed)
    tensors, targets, groups = ranker_dataset(named_images)
    validation_mask = np.isin(groups, validation_groups)
    training_mask = ~validation_mask
    if not training_mask.any() or not validation_mask.any():
        raise ValueError("Training and validation groups must both be non-empty")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(tensors[training_mask]),
            torch.from_numpy(targets[training_mask]),
        ),
        batch_size=128,
        shuffle=True,
    )
    model = CandidateCostCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_function = nn.SmoothL1Loss()
    for _ in range(epochs):
        model.train()
        for features, target in train_loader:
            prediction = model(features)
            loss = loss_function(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_predictions = model(
            torch.from_numpy(tensors[validation_mask])
        ).numpy()
    validation_targets = targets[validation_mask]
    correlation = pearsonr(validation_targets, validation_predictions).statistic
    metrics = RankerMetrics(
        validation_mae=float(
            np.mean(np.abs(validation_targets - validation_predictions))
        ),
        validation_pearson=float(correlation),
        training_samples=int(training_mask.sum()),
        validation_samples=int(validation_mask.sum()),
    )

    final_loader = DataLoader(
        TensorDataset(torch.from_numpy(tensors), torch.from_numpy(targets)),
        batch_size=128,
        shuffle=True,
    )
    final_model = CandidateCostCNN()
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3)
    for _ in range(epochs):
        final_model.train()
        for features, target in final_loader:
            prediction = final_model(features)
            loss = loss_function(prediction, target)
            final_optimizer.zero_grad()
            loss.backward()
            final_optimizer.step()
    final_model.eval()
    return CandidateRanker(final_model), CandidateRanker(model), metrics
