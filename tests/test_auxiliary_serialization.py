import numpy as np
import pytest

from abm_rdh import (
    AuxiliaryData,
    auxiliary_bits,
    capacity_bits,
    deserialize_auxiliary,
    embed,
    extract,
    optimize_tables_for_net_capacity,
    serialize_auxiliary,
)


def test_base_auxiliary_wire_round_trip() -> None:
    image = np.zeros((18, 18), dtype=np.uint8)
    image[0:3, 0:3] = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8
    )
    image[3:6, 0:3] = image[0:3, 0:3]
    result = embed(image, [1], policy="hamming1")
    wire = serialize_auxiliary(result.auxiliary)
    decoded = deserialize_auxiliary(wire, image_shape=image.shape)
    restored, recovered = extract(result.stego, decoded)
    assert recovered == [1]
    assert np.array_equal(restored, image)
    assert auxiliary_bits(result.auxiliary) == 8 * len(wire)


def test_net_optimizer_never_returns_negative_capacity() -> None:
    image = np.zeros((18, 18), dtype=np.uint8)
    image[0:3, 0:3] = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8
    )
    image[3:6, 0:3] = image[0:3, 0:3]
    result = embed(image, [1], policy="hamming1")
    optimized = optimize_tables_for_net_capacity(
        image,
        result.auxiliary.mapping_tables,
        policy="hamming1",
    )
    if optimized:
        gross = capacity_bits(image, optimized)
        auxiliary = AuxiliaryData(
            mapping_tables=optimized,
            payload_length=gross,
            image_shape=image.shape,
            policy="hamming1",
        )
        assert gross - auxiliary_bits(auxiliary) > 0


@pytest.mark.parametrize("table_count", [0, 1, 7, 25, 80])
def test_enumerative_hamming1_codec_round_trip(table_count: int) -> None:
    peaks = list(range(0, min(512, table_count * 5), 5))[:table_count]
    tables = {
        peak: {0: peak, 1: peak ^ (1 << (peak % 9))}
        for peak in peaks
    }
    auxiliary = AuxiliaryData(
        mapping_tables=tables,
        payload_length=1234,
        image_shape=(512, 512),
        policy="hamming1",
    )
    decoded = deserialize_auxiliary(
        serialize_auxiliary(auxiliary),
        image_shape=(512, 512),
    )
    assert decoded.mapping_tables == tables
    assert decoded.payload_length == auxiliary.payload_length
