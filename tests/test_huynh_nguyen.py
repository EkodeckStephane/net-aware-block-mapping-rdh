from pathlib import Path

import numpy as np
from PIL import Image

from abm_rdh import as_binary
from huynh_nguyen import (
    auxiliary_bits,
    capacity_bits,
    deserialize_auxiliary,
    embed,
    extract,
    serialize_auxiliary,
)
from run_experiments import DEFAULT_IMAGES, DOCUMENT_IMAGE_DIR


REFERENCE_CAPACITIES = {
    "circuit-2.png": 976,
    "formula-5.png": 2320,
    "graph1-5.png": 2448,
    "handwr2-8.png": 1176,
    "large-8.png": 1456,
    "symbol-6.png": 2704,
    "table1-3.png": 3066,
    "french-4.png": 2415,
}


def load_image(filename: str) -> np.ndarray:
    with Image.open(DOCUMENT_IMAGE_DIR / filename) as image:
        return as_binary(np.asarray(image.convert("L")))


def test_capacities_reproduce_published_reference_table() -> None:
    assert set(REFERENCE_CAPACITIES) == set(DEFAULT_IMAGES)
    for filename, expected in REFERENCE_CAPACITIES.items():
        assert capacity_bits(load_image(filename)) == expected


def test_huynh_round_trip_and_wire_format() -> None:
    image = load_image("circuit-2.png")
    message = np.random.default_rng(12).integers(
        0,
        2,
        size=capacity_bits(image),
        dtype=np.uint8,
    ).tolist()
    result = embed(image, message)
    wire = serialize_auxiliary(result.auxiliary)
    restored, recovered = extract(result.stego, deserialize_auxiliary(wire))
    assert recovered == message
    assert np.array_equal(restored, image)
    assert auxiliary_bits(result.auxiliary) == 8 * len(wire)
