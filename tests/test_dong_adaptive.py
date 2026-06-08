import numpy as np
from pathlib import Path

from run_experiments import DOCUMENT_IMAGE_DIR, load_binary

from dong_adaptive import (
    auxiliary_bits,
    deserialize_auxiliary,
    difference_matrix,
    embed,
    extract,
    inverse_difference,
    serialize_auxiliary,
)


def test_difference_matrix_round_trip() -> None:
    rng = np.random.default_rng(21)
    image = rng.integers(0, 2, size=(20, 21), dtype=np.uint8)
    assert np.array_equal(inverse_difference(difference_matrix(image)), image)


def test_dong_conservative_round_trip() -> None:
    image = load_binary(DOCUMENT_IMAGE_DIR / "circuit-2.png")
    message = np.random.default_rng(22).integers(
        0,
        2,
        size=100,
        dtype=np.uint8,
    ).tolist()
    result = embed(image, message, context_divisor=10)
    assert len(result.auxiliary.location_map) > 0
    wire = serialize_auxiliary(result.auxiliary)
    restored, recovered = extract(result.stego, deserialize_auxiliary(wire))
    assert recovered == message
    assert np.array_equal(restored, image)
    assert auxiliary_bits(result.auxiliary) == 8 * len(wire)
