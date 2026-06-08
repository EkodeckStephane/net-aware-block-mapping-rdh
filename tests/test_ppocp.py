import numpy as np

from ppocp import (
    COUNTS_BY_LEVEL,
    LEVEL_BY_COUNTS,
    auxiliary_bits,
    deserialize_auxiliary,
    embed,
    extract,
    fit_profile,
    level_map,
    pair_positions,
    serialize_auxiliary,
)


def test_all_25_levels_have_opposite_pairs() -> None:
    assert set(LEVEL_BY_COUNTS.values()) == set(range(1, 26))
    for level, (different_nn, different_fn) in COUNTS_BY_LEVEL.items():
        opposite = LEVEL_BY_COUNTS[(4 - different_nn, 4 - different_fn)]
        assert level + opposite == 26


def test_level_map_identifies_uniform_center() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    assert level_map(image)[2, 2] == 1
    image[2, 2] = 1
    assert level_map(image)[2, 2] == 25


def test_self_opposite_pair_is_not_double_counted() -> None:
    rng = np.random.default_rng(9)
    image = rng.integers(0, 2, size=(40, 40), dtype=np.uint8)
    levels = level_map(image)
    assert pair_positions(image, 13).size == np.count_nonzero(levels == 13)


def test_round_trip_with_serialized_auxiliary() -> None:
    rng = np.random.default_rng(20260608)
    training = [rng.integers(0, 2, size=(32, 32), dtype=np.uint8) for _ in range(3)]
    profile = fit_profile(training)
    image = rng.integers(0, 2, size=(32, 32), dtype=np.uint8)
    message = rng.integers(0, 2, size=100, dtype=np.uint8).tolist()
    result = embed(image, message, profile)
    wire = serialize_auxiliary(result.auxiliary)
    restored, recovered = extract(result.stego, deserialize_auxiliary(wire))
    assert recovered == message
    assert np.array_equal(restored, image)
    assert auxiliary_bits(result.auxiliary) == 8 * len(wire)
