"""Reimplementation of Yin et al.'s PPOCP reversible data hiding scheme.

The paper leaves the overlapping-block synchronization and location-map codec
underspecified. This implementation stores the used center positions and their
original values in a compact auxiliary stream. That conservative choice makes
extraction and restoration exact even when neighboring overlapping patterns
change during embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import struct
from typing import Iterable, Sequence
import zlib

import numpy as np
from scipy.signal import convolve2d

from abm_rdh import DRD_WEIGHTS, as_binary


_MAGIC = b"PPOC"
_VERSION = 1
_HEADER = ">4sBII B I I"


def _build_level_tables() -> tuple[dict[tuple[int, int], int], dict[int, tuple[int, int]]]:
    patterns = []
    for different_nn in range(5):
        for different_fn in range(5):
            distance = 3.4142 - different_nn - different_fn / sqrt(2.0)
            patterns.append((distance, different_nn, different_fn))
    patterns.sort(key=lambda item: (-item[0], item[1], item[2]))
    by_counts = {
        (different_nn, different_fn): level
        for level, (_, different_nn, different_fn) in enumerate(patterns, start=1)
    }
    by_level = {level: counts for counts, level in by_counts.items()}
    return by_counts, by_level


LEVEL_BY_COUNTS, COUNTS_BY_LEVEL = _build_level_tables()


@dataclass(frozen=True)
class PPOCPProfile:
    """Training-set average single-pixel distortion for the 25 levels."""

    single_flip_cost: tuple[float, ...]
    training_images: int

    def cost(self, level: int) -> float:
        if not 1 <= level <= 25:
            raise ValueError("PPOCP level must be in [1, 25]")
        return self.single_flip_cost[level - 1]


@dataclass(frozen=True)
class PPOCPAuxiliary:
    image_shape: tuple[int, int]
    first_level: int
    payload_length: int
    positions: tuple[int, ...]
    original_centers: tuple[int, ...]

    @property
    def second_level(self) -> int:
        return 26 - self.first_level


@dataclass(frozen=True)
class PPOCPEmbeddingResult:
    stego: np.ndarray
    auxiliary: PPOCPAuxiliary
    embedded_bits: int
    changed_pixels: int


def level_map(image: np.ndarray) -> np.ndarray:
    """Return PPOCP levels for all valid 3x3 centers, zero on the border."""

    binary = as_binary(image)
    levels = np.zeros(binary.shape, dtype=np.uint8)
    center = binary[1:-1, 1:-1]
    different_nn = sum(
        comparison.astype(np.uint8)
        for comparison in (
            binary[:-2, 1:-1] != center,
            binary[2:, 1:-1] != center,
            binary[1:-1, :-2] != center,
            binary[1:-1, 2:] != center,
        )
    )
    different_fn = sum(
        comparison.astype(np.uint8)
        for comparison in (
            binary[:-2, :-2] != center,
            binary[:-2, 2:] != center,
            binary[2:, :-2] != center,
            binary[2:, 2:] != center,
        )
    )
    interior = np.empty(center.shape, dtype=np.uint8)
    for counts, level in LEVEL_BY_COUNTS.items():
        interior[(different_nn == counts[0]) & (different_fn == counts[1])] = level
    levels[1:-1, 1:-1] = interior
    return levels


def fit_profile(images: Iterable[np.ndarray]) -> PPOCPProfile:
    """Estimate Eq. (11) on a training set without modifying the images."""

    sums = np.zeros(25, dtype=np.float64)
    counts = np.zeros(25, dtype=np.int64)
    image_count = 0
    for image in images:
        binary = as_binary(image)
        levels = level_map(binary)
        weighted_ones = convolve2d(
            binary.astype(np.float64),
            DRD_WEIGHTS,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        valid_weight = convolve2d(
            np.ones(binary.shape, dtype=np.float64),
            DRD_WEIGHTS,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        flip_cost = np.where(binary == 0, valid_weight - weighted_ones, weighted_ones)
        for level in range(1, 26):
            mask = levels == level
            if np.any(mask):
                sums[level - 1] += float(flip_cost[mask].sum())
                counts[level - 1] += int(mask.sum())
        image_count += 1
    if image_count == 0:
        raise ValueError("At least one training image is required")
    costs = np.divide(sums, counts, out=np.full(25, np.inf), where=counts > 0)
    return PPOCPProfile(tuple(float(value) for value in costs), image_count)


def pair_positions(image: np.ndarray, first_level: int) -> np.ndarray:
    """Return flattened center positions belonging to one opposite pair."""

    if not 1 <= first_level <= 13:
        raise ValueError("The canonical first PPOCP level must be in [1, 13]")
    levels = level_map(image)
    second_level = 26 - first_level
    mask = (levels == first_level) | (levels == second_level)
    return np.flatnonzero(mask)


def _encode_varints(values: Sequence[int]) -> bytes:
    output = bytearray()
    previous = 0
    for index, value in enumerate(values):
        delta = int(value) if index == 0 else int(value) - previous
        previous = int(value)
        while delta >= 0x80:
            output.append((delta & 0x7F) | 0x80)
            delta >>= 7
        output.append(delta)
    return bytes(output)


def _decode_varints(payload: bytes, count: int) -> tuple[list[int], int]:
    values: list[int] = []
    offset = 0
    previous = 0
    while len(values) < count:
        shift = 0
        delta = 0
        while True:
            if offset >= len(payload):
                raise ValueError("Truncated PPOCP position stream")
            byte = payload[offset]
            offset += 1
            delta |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        value = delta if not values else previous + delta
        values.append(value)
        previous = value
    return values, offset


def serialize_auxiliary(auxiliary: PPOCPAuxiliary) -> bytes:
    count = len(auxiliary.positions)
    if len(auxiliary.original_centers) != count:
        raise ValueError("PPOCP positions and original centers have different lengths")
    packed_centers = np.packbits(
        np.asarray(auxiliary.original_centers, dtype=np.uint8),
        bitorder="big",
    ).tobytes()
    raw = struct.pack(
        _HEADER,
        _MAGIC,
        _VERSION,
        int(auxiliary.image_shape[0]),
        int(auxiliary.image_shape[1]),
        int(auxiliary.first_level),
        int(auxiliary.payload_length),
        count,
    )
    raw += _encode_varints(auxiliary.positions) + packed_centers
    compressed = zlib.compress(raw, level=9)
    return b"\x01" + compressed if len(compressed) < len(raw) else b"\x00" + raw


def deserialize_auxiliary(payload: bytes) -> PPOCPAuxiliary:
    if not payload:
        raise ValueError("Empty PPOCP auxiliary payload")
    if payload[0] == 0:
        raw = payload[1:]
    elif payload[0] == 1:
        raw = zlib.decompress(payload[1:])
    else:
        raise ValueError("Unknown PPOCP compression marker")
    header_size = struct.calcsize(_HEADER)
    magic, version, height, width, first_level, payload_length, count = struct.unpack_from(
        _HEADER, raw, 0
    )
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("Unsupported PPOCP auxiliary format")
    positions, consumed = _decode_varints(raw[header_size:], count)
    packed = raw[header_size + consumed :]
    centers = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8),
        bitorder="big",
    )[:count]
    if centers.size != count:
        raise ValueError("Truncated PPOCP center stream")
    return PPOCPAuxiliary(
        image_shape=(int(height), int(width)),
        first_level=int(first_level),
        payload_length=int(payload_length),
        positions=tuple(int(value) for value in positions),
        original_centers=tuple(int(value) for value in centers),
    )


def auxiliary_bits(auxiliary: PPOCPAuxiliary) -> int:
    return 8 * len(serialize_auxiliary(auxiliary))


def _paper_score(
    image: np.ndarray,
    profile: PPOCPProfile,
    first_level: int,
    payload_length: int | None,
) -> float:
    levels = level_map(image)
    second_level = 26 - first_level
    first_count = int(np.count_nonzero(levels == first_level))
    second_count = int(np.count_nonzero(levels == second_level))
    capacity = first_count if first_level == second_level else first_count + second_count
    if capacity == 0 or (payload_length is not None and capacity < payload_length):
        return -np.inf
    used = capacity if payload_length is None else payload_length
    if first_level == second_level:
        distortion = used * profile.cost(first_level) / 2.0
    else:
        expected_first = used * first_count / capacity / 2.0
        expected_second = used * second_count / capacity / 2.0
        distortion = (
            expected_first * profile.cost(first_level)
            + expected_second * profile.cost(second_level)
        )
    return used / max(distortion, np.finfo(float).eps)


def select_pair(
    image: np.ndarray,
    profile: PPOCPProfile,
    *,
    payload_length: int | None = None,
) -> int:
    """Select the canonical first level using the paper's balanced principle."""

    scored = [
        (_paper_score(image, profile, first_level, payload_length), first_level)
        for first_level in range(1, 14)
    ]
    score, first_level = max(scored, key=lambda item: (item[0], -item[1]))
    if not np.isfinite(score):
        raise ValueError("No PPOCP pair can carry the requested payload")
    return first_level


def capacity_bits(
    image: np.ndarray,
    profile: PPOCPProfile,
    *,
    first_level: int | None = None,
) -> tuple[int, int]:
    selected = first_level or select_pair(image, profile)
    return int(pair_positions(image, selected).size), selected


def embed(
    image: np.ndarray,
    message_bits: Iterable[int],
    profile: PPOCPProfile,
    *,
    first_level: int | None = None,
) -> PPOCPEmbeddingResult:
    original = as_binary(image)
    message = np.asarray([int(bit) for bit in message_bits], dtype=np.uint8)
    if np.any((message != 0) & (message != 1)):
        raise ValueError("Message bits must contain only zero and one")
    selected = first_level or select_pair(
        original,
        profile,
        payload_length=int(message.size),
    )
    available = pair_positions(original, selected)
    if message.size > available.size:
        raise ValueError(
            f"Payload has {message.size} bits but PPOCP capacity is {available.size}"
        )
    used = available[: message.size]
    flat_original = original.reshape(-1)
    original_centers = flat_original[used].copy()
    stego = original.copy()
    stego.reshape(-1)[used] = message
    auxiliary = PPOCPAuxiliary(
        image_shape=original.shape,
        first_level=selected,
        payload_length=int(message.size),
        positions=tuple(int(value) for value in used),
        original_centers=tuple(int(value) for value in original_centers),
    )
    return PPOCPEmbeddingResult(
        stego=stego,
        auxiliary=auxiliary,
        embedded_bits=int(message.size),
        changed_pixels=int(np.count_nonzero(original_centers != message)),
    )


def extract(
    stego: np.ndarray,
    auxiliary: PPOCPAuxiliary,
) -> tuple[np.ndarray, list[int]]:
    binary = as_binary(stego)
    if binary.shape != auxiliary.image_shape:
        raise ValueError("Stego image shape does not match PPOCP auxiliary data")
    positions = np.asarray(auxiliary.positions, dtype=np.int64)
    if positions.size != auxiliary.payload_length:
        raise ValueError("PPOCP auxiliary position count does not match payload length")
    recovered = binary.reshape(-1)[positions].astype(np.uint8).tolist()
    restored = binary.copy()
    restored.reshape(-1)[positions] = np.asarray(
        auxiliary.original_centers,
        dtype=np.uint8,
    )
    return restored, recovered
