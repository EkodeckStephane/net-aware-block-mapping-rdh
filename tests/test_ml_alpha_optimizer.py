import numpy as np

from ml_alpha_optimizer import best_observation, expected_improvement, optimize_alpha


class DummyModel:
    def predict(self, values, return_std=False):
        mean = (values[:, 0] - 0.5) ** 2
        std = np.full_like(mean, 0.1)
        return (mean, std) if return_std else mean


def test_expected_improvement_is_finite() -> None:
    candidates = np.linspace(0.0, 1.0, 11)
    values = expected_improvement(DummyModel(), candidates, best_value=0.1)
    assert values.shape == candidates.shape
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)


def test_optimizer_returns_unique_bounded_observations() -> None:
    image = np.zeros((12, 12), dtype=np.uint8)
    image[3:9, 3:9] = np.indices((6, 6)).sum(axis=0) % 2
    observations = optimize_alpha(
        image,
        alpha_bounds=(0.0, 1.0),
        initial_alphas=(0.0, 0.5, 1.0),
        iterations=2,
        grid_size=21,
    )
    alphas = [item.alpha for item in observations]
    assert len(alphas) == len(set(alphas))
    assert all(0.0 <= alpha <= 1.0 for alpha in alphas)
    assert best_observation(observations) in observations
