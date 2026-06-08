"""Gaussian-process Bayesian optimization of the adaptive threshold alpha."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from abm_rdh import build_mapping_tables, capacity_bits, evaluate


@dataclass(frozen=True)
class AlphaObservation:
    alpha: float
    capacity_bits: int
    psnr_db: float
    drd: float
    objective: float


def _normal_pdf(value: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * value**2) / sqrt(2.0 * pi)


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(value / sqrt(2.0)))


def expected_improvement(
    model: GaussianProcessRegressor,
    candidates: np.ndarray,
    best_value: float,
    *,
    exploration: float = 0.01,
) -> np.ndarray:
    mean, std = model.predict(candidates.reshape(-1, 1), return_std=True)
    improvement = best_value - mean - exploration
    safe_std = np.maximum(std, 1e-12)
    z_score = improvement / safe_std
    acquisition = improvement * _normal_cdf(z_score) + safe_std * _normal_pdf(
        z_score
    )
    acquisition[std <= 1e-12] = 0.0
    return acquisition


def evaluate_alpha(
    image: np.ndarray,
    alpha: float,
    message_bits: list[int],
    *,
    capacity_scale: int,
    lambda_weight: float,
) -> AlphaObservation:
    tables = build_mapping_tables(image, alpha=alpha, policy="adaptive")
    available = capacity_bits(image, tables)
    payload = message_bits[:available]
    _, _, _, metrics = evaluate(
        image,
        payload,
        alpha=alpha,
        policy="adaptive",
    )
    normalized_capacity_penalty = 1.0 - available / max(capacity_scale, 1)
    objective = (
        lambda_weight * normalized_capacity_penalty
        + (1.0 - lambda_weight) * float(metrics["drd"])
    )
    return AlphaObservation(
        alpha=float(alpha),
        capacity_bits=available,
        psnr_db=float(metrics["psnr_db"]),
        drd=float(metrics["drd"]),
        objective=float(objective),
    )


def optimize_alpha(
    image: np.ndarray,
    *,
    seed: int = 20260607,
    alpha_bounds: tuple[float, float] = (0.0, 1.5),
    initial_alphas: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5),
    iterations: int = 6,
    lambda_weight: float = 0.5,
    grid_size: int = 151,
) -> list[AlphaObservation]:
    """Minimize a normalized capacity--DRD objective with GP/EI."""

    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError("lambda_weight must be in [0, 1]")
    lower, upper = alpha_bounds
    if lower >= upper:
        raise ValueError("alpha_bounds must be increasing")

    rng = np.random.default_rng(seed)
    reference_tables = build_mapping_tables(image, alpha=lower, policy="adaptive")
    reference_capacity = capacity_bits(image, reference_tables)
    message = rng.integers(0, 2, size=max(image.size, 1), dtype=np.uint8).tolist()

    cache: dict[float, AlphaObservation] = {}

    def observe(alpha: float) -> AlphaObservation:
        key = round(float(alpha), 6)
        if key not in cache:
            cache[key] = evaluate_alpha(
                image,
                key,
                message,
                capacity_scale=reference_capacity,
                lambda_weight=lambda_weight,
            )
        return cache[key]

    for alpha in initial_alphas:
        if lower <= alpha <= upper:
            observe(alpha)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=0.3,
        length_scale_bounds=(1e-2, 10.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-2))

    candidate_grid = np.linspace(lower, upper, grid_size)
    for _ in range(iterations):
        observations = sorted(cache.values(), key=lambda item: item.alpha)
        x_train = np.array([item.alpha for item in observations]).reshape(-1, 1)
        y_train = np.array([item.objective for item in observations])
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=2,
        )
        model.fit(x_train, y_train)
        acquisition = expected_improvement(model, candidate_grid, y_train.min())
        for observed_alpha in cache:
            acquisition[np.isclose(candidate_grid, observed_alpha)] = -np.inf
        next_alpha = float(candidate_grid[int(np.argmax(acquisition))])
        observe(next_alpha)

    return sorted(cache.values(), key=lambda item: item.alpha)


def best_observation(observations: list[AlphaObservation]) -> AlphaObservation:
    if not observations:
        raise ValueError("At least one observation is required")
    return min(observations, key=lambda item: (item.objective, item.alpha))
