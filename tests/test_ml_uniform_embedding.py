import numpy as np

from ml_uniform_agent import UniformBlockAgent
from ml_uniform_embedding import embed_uniform_bits, extract_uniform_bits
from ml_uniform_embedding import (
    deserialize_uniform_auxiliary,
    serialize_uniform_auxiliary,
    uniform_auxiliary_bits,
)


class SafeModel:
    def predict_proba(self, features):
        return np.column_stack(
            [np.zeros(len(features), dtype=float), np.ones(len(features), dtype=float)]
        )


class PositionModel:
    def predict(self, features):
        return np.zeros(len(features), dtype=np.uint8)


def test_uniform_embedding_round_trip() -> None:
    image = np.zeros((12, 12), dtype=np.uint8)
    agent = UniformBlockAgent(SafeModel(), PositionModel())
    message = [1, 0, 1]
    stego, auxiliary = embed_uniform_bits(image, message, agent)
    wire = serialize_uniform_auxiliary(auxiliary)
    decoded = deserialize_uniform_auxiliary(wire)
    restored, recovered = extract_uniform_bits(stego, decoded, agent)
    assert recovered == message
    assert np.array_equal(restored, image)
    assert uniform_auxiliary_bits(auxiliary) == len(wire) * 8
    assert uniform_auxiliary_bits(auxiliary) <= len(message) * 64
