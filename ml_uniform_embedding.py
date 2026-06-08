"""Reversible uniform-block embedding with measured compact side information."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable
import zlib

import numpy as np

from abm_rdh import as_binary
from ml_uniform_agent import UniformBlockAgent, find_safe_uniform_blocks


_UNIFORM_MAGIC = b"ABMU"


@dataclass(frozen=True)
class UniformLocation:
    row: int
    col: int
    flip_position: int


@dataclass(frozen=True)
class UniformAuxiliaryData:
    locations: tuple[UniformLocation, ...]
    payload_length: int
    image_shape: tuple[int, int]
    probability_threshold: float


def _encode_varint(value: int) -> bytes:
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _decode_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def serialize_uniform_auxiliary(auxiliary: UniformAuxiliaryData) -> bytes:
    threshold = int(round(auxiliary.probability_threshold * 65535.0))
    block_columns = auxiliary.image_shape[1] // 3
    body = bytearray()
    previous = 0
    for index, location in enumerate(auxiliary.locations):
        linear = (location.row // 3) * block_columns + location.col // 3
        delta = linear if index == 0 else linear - previous
        body.extend(_encode_varint(delta))
        body.append(int(location.flip_position))
        previous = linear
    raw = struct.pack(
        ">4sBIHHHH",
        _UNIFORM_MAGIC,
        1,
        int(auxiliary.payload_length),
        int(auxiliary.image_shape[0]),
        int(auxiliary.image_shape[1]),
        threshold,
        len(auxiliary.locations),
    ) + bytes(body)
    compressed = zlib.compress(raw, level=9)
    return (b"\x01" + compressed) if len(compressed) < len(raw) else (b"\x00" + raw)


def deserialize_uniform_auxiliary(payload: bytes) -> UniformAuxiliaryData:
    raw = zlib.decompress(payload[1:]) if payload[0] == 1 else payload[1:]
    header_format = ">4sBIHHHH"
    header_size = struct.calcsize(header_format)
    magic, version, length, height, width, threshold, count = struct.unpack_from(
        header_format, raw, 0
    )
    if magic != _UNIFORM_MAGIC or version != 1:
        raise ValueError("Unsupported uniform auxiliary format")
    block_columns = width // 3
    offset = header_size
    previous = 0
    locations = []
    for index in range(count):
        delta, offset = _decode_varint(raw, offset)
        linear = delta if index == 0 else previous + delta
        position = raw[offset]
        offset += 1
        locations.append(
            UniformLocation(
                row=(linear // block_columns) * 3,
                col=(linear % block_columns) * 3,
                flip_position=int(position),
            )
        )
        previous = linear
    return UniformAuxiliaryData(
        locations=tuple(locations),
        payload_length=int(length),
        image_shape=(int(height), int(width)),
        probability_threshold=float(threshold) / 65535.0,
    )


def uniform_auxiliary_bits(auxiliary: UniformAuxiliaryData) -> int:
    return 8 * len(serialize_uniform_auxiliary(auxiliary))


def uniform_capacity(
    image: np.ndarray,
    agent: UniformBlockAgent,
    *,
    probability_threshold: float = 0.7,
) -> int:
    return len(
        find_safe_uniform_blocks(
            image, agent, probability_threshold=probability_threshold
        )
    )


def embed_uniform_bits(
    image: np.ndarray,
    message_bits: Iterable[int],
    agent: UniformBlockAgent,
    *,
    probability_threshold: float = 0.7,
) -> tuple[np.ndarray, UniformAuxiliaryData]:
    binary = as_binary(image)
    message = [int(bit) for bit in message_bits]
    if any(bit not in (0, 1) for bit in message):
        raise ValueError("Message bits must contain only zero and one")
    threshold = round(probability_threshold * 65535.0) / 65535.0
    candidates = find_safe_uniform_blocks(
        binary, agent, probability_threshold=threshold
    )
    if len(message) > len(candidates):
        raise ValueError(
            f"Uniform payload has {len(message)} bits but capacity is {len(candidates)}"
        )
    stego = binary.copy()
    locations = []
    for bit, candidate in zip(message, candidates):
        locations.append(
            UniformLocation(
                row=candidate.row,
                col=candidate.col,
                flip_position=candidate.flip_position,
            )
        )
        if bit:
            stego[
                candidate.row + candidate.flip_position // 3,
                candidate.col + candidate.flip_position % 3,
            ] ^= 1
    return stego, UniformAuxiliaryData(
        locations=tuple(locations),
        payload_length=len(message),
        image_shape=binary.shape,
        probability_threshold=threshold,
    )


def extract_uniform_bits(
    stego: np.ndarray,
    auxiliary: UniformAuxiliaryData,
    agent: UniformBlockAgent | None = None,
) -> tuple[np.ndarray, list[int]]:
    del agent
    binary = as_binary(stego)
    if binary.shape != auxiliary.image_shape:
        raise ValueError("Stego image shape does not match uniform auxiliary data")
    restored = binary.copy()
    recovered = []
    for location in auxiliary.locations[: auxiliary.payload_length]:
        block = binary[
            location.row : location.row + 3,
            location.col : location.col + 3,
        ]
        original_value = int(block.sum() >= 5)
        pixel_row = location.row + location.flip_position // 3
        pixel_col = location.col + location.flip_position % 3
        recovered.append(int(binary[pixel_row, pixel_col] != original_value))
        restored[
            location.row : location.row + 3,
            location.col : location.col + 3,
        ] = original_value
    return restored, recovered
