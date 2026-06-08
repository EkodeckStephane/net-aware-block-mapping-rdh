import numpy as np
from PIL import Image

from evaluate_bossbase_net import load_thresholded


def test_bossbase_binarization_uses_explicit_threshold(tmp_path) -> None:
    path = tmp_path / "sample.pgm"
    Image.fromarray(np.array([[0, 127, 128, 255]], dtype=np.uint8)).save(path)
    binary = load_thresholded(path, 128)
    assert binary.tolist() == [[0, 0, 1, 1]]
