from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from abm_rdh import (
    as_binary,
    build_mapping_tables,
    capacity_bits,
    drd,
    embed,
    extract,
    pattern_array,
    pattern_index,
    psnr,
)
from run_experiments import DEFAULT_IMAGES, DOCUMENT_IMAGE_DIR


IMAGE_DIR = DOCUMENT_IMAGE_DIR


def load_image(filename: str) -> np.ndarray:
    with Image.open(IMAGE_DIR / filename) as image:
        return as_binary(np.asarray(image.convert("L")))


@pytest.mark.parametrize("pattern", [0, 1, 7, 255, 256, 510, 511])
def test_pattern_round_trip(pattern: int) -> None:
    assert pattern_index(pattern_array(pattern)) == pattern


def test_metrics_have_known_simple_values() -> None:
    original = np.zeros((8, 8), dtype=np.uint8)
    stego = original.copy()
    stego[4, 4] = 1
    assert psnr(original, stego) == pytest.approx(10 * np.log10(64))
    assert drd(original, original) == 0.0
    assert drd(original, stego) > 0.0


@pytest.mark.parametrize("filename", DEFAULT_IMAGES)
def test_full_capacity_round_trip(filename: str) -> None:
    image = load_image(filename)
    tables = build_mapping_tables(image, alpha=0.1)
    capacity = capacity_bits(image, tables)
    message = np.random.default_rng(1234).integers(
        0, 2, size=capacity, dtype=np.uint8
    ).tolist()

    result = embed(image, message, alpha=0.1)
    restored, recovered = extract(result.stego, result.auxiliary)

    assert result.embedded_bits == capacity
    assert recovered == message
    assert np.array_equal(restored, image)
    assert result.stego.shape == image.shape


@pytest.mark.parametrize("filename", DEFAULT_IMAGES)
def test_hamming1_policy_round_trip(filename: str) -> None:
    image = load_image(filename)
    tables = build_mapping_tables(image, policy="hamming1")
    capacity = capacity_bits(image, tables)
    message = np.random.default_rng(4321).integers(
        0, 2, size=capacity, dtype=np.uint8
    ).tolist()
    result = embed(image, message, policy="hamming1")
    restored, recovered = extract(result.stego, result.auxiliary)
    assert recovered == message
    assert np.array_equal(restored, image)
    assert all(len(table) == 2 for table in tables.values())
    assert all(
        (peak ^ replacement).bit_count() == 1
        for peak, table in tables.items()
        for symbol, replacement in table.items()
        if symbol == 1
    )


def test_embedding_is_deterministic() -> None:
    image = load_image(DEFAULT_IMAGES[0])
    tables = build_mapping_tables(image, alpha=0.1)
    capacity = capacity_bits(image, tables)
    message = [index % 2 for index in range(capacity)]
    first = embed(image, message, alpha=0.1)
    second = embed(image, message, alpha=0.1)
    assert np.array_equal(first.stego, second.stego)
    assert first.auxiliary == second.auxiliary


def test_payload_over_capacity_is_rejected() -> None:
    image = load_image(DEFAULT_IMAGES[0])
    tables = build_mapping_tables(image, alpha=0.1)
    capacity = capacity_bits(image, tables)
    with pytest.raises(ValueError, match="capacity"):
        embed(image, [0] * (capacity + 1), alpha=0.1)


def test_partial_payload_is_not_silently_truncated() -> None:
    image = load_image(DEFAULT_IMAGES[0])
    message = [1, 0, 1, 1, 0]
    result = embed(image, message, alpha=0.1)
    restored, recovered = extract(result.stego, result.auxiliary)
    assert recovered == message
    assert result.embedded_bits == len(message)
    assert np.array_equal(restored, image)


def test_mapping_tables_are_globally_injective() -> None:
    image = load_image(DEFAULT_IMAGES[0])
    tables = build_mapping_tables(image, policy="hamming1")
    states = [
        pattern
        for table in tables.values()
        for pattern in table.values()
    ]
    assert len(states) == len(set(states))


def test_incomplete_border_pixels_remain_unchanged() -> None:
    image = np.zeros((8, 11), dtype=np.uint8)
    peak = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    image[0:3, 0:3] = peak
    image[3:6, 0:3] = peak
    image[6:, :] = 1
    image[:, 9:] = 1

    result = embed(image, [1], policy="hamming1")
    restored, recovered = extract(result.stego, result.auxiliary)

    assert recovered == [1]
    assert np.array_equal(result.stego[6:, :], image[6:, :])
    assert np.array_equal(result.stego[:, 9:], image[:, 9:])
    assert np.array_equal(restored, image)
