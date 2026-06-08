"""Huynh--Nguyen fixed-threshold block-mapping baseline.

The implementation reproduces the capacities reported for the eight reference
images with the paper's 3x3 non-uniform PEAK, Hamming threshold T=5, and
nearest ZERO selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log2
import struct
from typing import Iterable
import zlib

import numpy as np

from abm_rdh import (
    UNIFORM_PATTERNS,
    as_binary,
    hamming,
    iter_blocks,
    pattern_array,
    pattern_index,
)


HAMMING_THRESHOLD = 5
_MAGIC = b"HNBM"
_VERSION = 1


@dataclass(frozen=True)
class HuynhAuxiliary:
    image_shape: tuple[int, int]
    peak: int
    states: tuple[int, ...]
    payload_length: int


@dataclass(frozen=True)
class HuynhEmbeddingResult:
    stego: np.ndarray
    auxiliary: HuynhAuxiliary
    embedded_bits: int
    modified_blocks: int


def build_mapping(image: np.ndarray) -> tuple[int, tuple[int, ...], int]:
    """Return the paper's PEAK, ordered states, and gross capacity."""

    binary = as_binary(image)
    frequencies: dict[int, int] = {}
    for _, _, block in iter_blocks(binary):
        pattern = pattern_index(block)
        if pattern not in UNIFORM_PATTERNS:
            frequencies[pattern] = frequencies.get(pattern, 0) + 1
    if not frequencies:
        return -1, (), 0
    peak = min(
        frequencies,
        key=lambda pattern: (-frequencies[pattern], pattern),
    )
    present = {
        pattern_index(block)
        for _, _, block in iter_blocks(binary)
    }
    candidates = sorted(
        (
            (hamming(peak, zero), zero)
            for zero in range(512)
            if zero not in present and hamming(peak, zero) <= HAMMING_THRESHOLD
        ),
        key=lambda item: (item[0], item[1]),
    )
    bits_per_block = floor(log2(len(candidates) + 1))
    if bits_per_block == 0:
        return peak, (peak,), 0
    selected = tuple(zero for _, zero in candidates[: (1 << bits_per_block) - 1])
    states = (peak, *selected)
    return peak, states, frequencies[peak] * bits_per_block


def capacity_bits(image: np.ndarray) -> int:
    return build_mapping(image)[2]


def serialize_auxiliary(auxiliary: HuynhAuxiliary) -> bytes:
    raw = struct.pack(
        ">4sBIIIHH",
        _MAGIC,
        _VERSION,
        auxiliary.image_shape[0],
        auxiliary.image_shape[1],
        auxiliary.payload_length,
        auxiliary.peak,
        len(auxiliary.states),
    )
    raw += struct.pack(f">{len(auxiliary.states)}H", *auxiliary.states)
    compressed = zlib.compress(raw, level=9)
    return b"\x01" + compressed if len(compressed) < len(raw) else b"\x00" + raw


def deserialize_auxiliary(payload: bytes) -> HuynhAuxiliary:
    if not payload:
        raise ValueError("Empty Huynh--Nguyen auxiliary payload")
    raw = payload[1:] if payload[0] == 0 else zlib.decompress(payload[1:])
    header_format = ">4sBIIIHH"
    header_size = struct.calcsize(header_format)
    magic, version, height, width, payload_length, peak, state_count = (
        struct.unpack_from(header_format, raw, 0)
    )
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("Unsupported Huynh--Nguyen auxiliary format")
    states = struct.unpack_from(f">{state_count}H", raw, header_size)
    return HuynhAuxiliary(
        image_shape=(int(height), int(width)),
        peak=int(peak),
        states=tuple(int(value) for value in states),
        payload_length=int(payload_length),
    )


def auxiliary_bits(auxiliary: HuynhAuxiliary) -> int:
    return 8 * len(serialize_auxiliary(auxiliary))


def embed(
    image: np.ndarray,
    message_bits: Iterable[int],
) -> HuynhEmbeddingResult:
    original = as_binary(image)
    message = [int(bit) for bit in message_bits]
    if any(bit not in (0, 1) for bit in message):
        raise ValueError("Message bits must contain only zero and one")
    peak, states, maximum = build_mapping(original)
    if len(message) > maximum:
        raise ValueError(f"Payload has {len(message)} bits but capacity is {maximum}")
    if not states:
        if message:
            raise ValueError("Image has no Huynh--Nguyen capacity")
        auxiliary = HuynhAuxiliary(original.shape, 0, (), 0)
        return HuynhEmbeddingResult(original.copy(), auxiliary, 0, 0)
    bits_per_block = int(log2(len(states)))
    stego = original.copy()
    offset = 0
    modified = 0
    for row, col, block in iter_blocks(original):
        if offset >= len(message):
            break
        if pattern_index(block) != peak:
            continue
        remaining = min(bits_per_block, len(message) - offset)
        chunk = message[offset : offset + remaining] + [0] * (
            bits_per_block - remaining
        )
        symbol = 0
        for bit in chunk:
            symbol = (symbol << 1) | bit
        replacement = states[symbol]
        stego[row : row + 3, col : col + 3] = pattern_array(replacement)
        modified += replacement != peak
        offset += remaining
    auxiliary = HuynhAuxiliary(
        image_shape=original.shape,
        peak=peak,
        states=states,
        payload_length=offset,
    )
    return HuynhEmbeddingResult(stego, auxiliary, offset, modified)


def extract(
    stego: np.ndarray,
    auxiliary: HuynhAuxiliary,
) -> tuple[np.ndarray, list[int]]:
    binary = as_binary(stego)
    if binary.shape != auxiliary.image_shape:
        raise ValueError("Stego shape does not match Huynh--Nguyen auxiliary data")
    if auxiliary.payload_length == 0:
        return binary.copy(), []
    bits_per_block = int(log2(len(auxiliary.states)))
    inverse = {
        pattern: symbol for symbol, pattern in enumerate(auxiliary.states)
    }
    restored = binary.copy()
    recovered: list[int] = []
    for row, col, block in iter_blocks(binary):
        if len(recovered) >= auxiliary.payload_length:
            break
        pattern = pattern_index(block)
        if pattern not in inverse:
            continue
        symbol = inverse[pattern]
        bits = [
            (symbol >> shift) & 1
            for shift in range(bits_per_block - 1, -1, -1)
        ]
        recovered.extend(bits[: auxiliary.payload_length - len(recovered)])
        restored[row : row + 3, col : col + 3] = pattern_array(auxiliary.peak)
    if len(recovered) != auxiliary.payload_length:
        raise ValueError("Incomplete Huynh--Nguyen payload")
    return restored, recovered
