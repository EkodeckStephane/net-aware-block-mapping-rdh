import numpy as np

from evaluate_steganalysis import features


def test_features_are_finite_and_fixed_length() -> None:
    image = np.zeros((8, 8), dtype=np.uint8)
    image[::2, ::2] = 1
    vector = features(image)
    assert vector.shape == (531,)
    assert np.all(np.isfinite(vector))
