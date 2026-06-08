import numpy as np

from ml_uniform_agent import (
    best_uniform_flip,
    context_features,
    flip_cost_map,
    local_flip_cost,
    uniform_training_samples,
)


def test_uniform_features_and_costs_are_well_formed() -> None:
    image = np.zeros((15, 15), dtype=np.uint8)
    image[6:9, 9:12] = 1
    features = context_features(image, 6, 6)
    position, cost = best_uniform_flip(image, 6, 6)
    assert features.shape == (85,)
    assert 0 <= position < 9
    assert 0.0 <= cost <= 1.0
    assert local_flip_cost(image, 6, 6) <= 1.0
    costs = flip_cost_map(image)
    assert np.isclose(costs[6, 6], local_flip_cost(image, 6, 6))


def test_training_samples_include_group_labels() -> None:
    first = np.zeros((15, 15), dtype=np.uint8)
    first[6:9, 9:12] = 1
    second = np.ones((15, 15), dtype=np.uint8)
    second[6:9, 9:12] = 0
    features, labels, positions, groups = uniform_training_samples(
        [("first", first), ("second", second)],
        safe_cost_threshold=0.8,
        unsafe_ratio=1,
    )
    assert len(features) == len(labels) == len(positions) == len(groups)
    assert set(groups) == {0, 1}
    assert features.shape[1] == 85
