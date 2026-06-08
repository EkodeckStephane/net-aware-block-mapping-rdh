import numpy as np
import torch

from ml_candidate_ranker import (
    CandidateCostCNN,
    candidate_tensor,
    empirical_replacement_cost,
)


def test_candidate_tensor_and_network_shape() -> None:
    context = np.zeros((9, 9), dtype=np.float32)
    tensor = candidate_tensor(7, 6, context)
    model = CandidateCostCNN()
    output = model(torch.from_numpy(tensor[None]))
    assert tensor.shape == (3, 9, 9)
    assert output.shape == (1,)
    assert output.item() >= 0.0


def test_empirical_cost_increases_with_more_flips() -> None:
    image = np.zeros((15, 15), dtype=np.uint8)
    image[6:9, 6:9] = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8
    )
    peak = 16
    positions = [(6, 6)]
    one_flip = empirical_replacement_cost(image, positions, peak, 0)
    two_flips = empirical_replacement_cost(image, positions, peak, 3)
    assert one_flip >= 0.0
    assert two_flips >= one_flip
