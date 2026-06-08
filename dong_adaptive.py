"""Conservative reimplementation of Dong et al.'s adaptive overlapping method.

The difference domain and adaptive (PM, PF, PFR) selection follow the paper.
The generated PF/PFR location map is compressed and serialized explicitly
instead of relying on boundary-ambiguous later embedding rounds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import struct
from typing import Iterable
import zlib

import numpy as np

from abm_rdh import as_binary


SUBSTITUTABLE = {
    0b0001: (0b0010, 0b0111, 0b1101),
    0b0010: (0b0001, 0b0100, 0b1110),
    0b0011: (0b0000, 0b0101, 0b1111),
    0b0100: (0b0010, 0b0111, 0b1000),
    0b0101: (0b0011, 0b0110, 0b1001),
    0b0110: (0b0000, 0b0101, 0b1010),
    0b0111: (0b0001, 0b0100, 0b1011),
    0b1000: (0b0100, 0b1011, 0b1110),
    0b1001: (0b1000, 0b1010, 0b1111),
    0b1010: (0b0110, 0b1001, 0b1100),
    0b1011: (0b1000, 0b0111, 0b1101),
    0b1100: (0b0000, 0b1010, 0b1111),
    0b1101: (0b0001, 0b1011, 0b1110),
    0b1110: (0b0010, 0b1000, 0b1101),
    0b1111: (0b0011, 0b1001, 0b1100),
}

_MAGIC = b"DOAP"
_VERSION = 1


@dataclass(frozen=True)
class DongAuxiliary:
    image_shape: tuple[int, int]
    context_divisor: int
    end_position: int
    secret_length: int
    location_map: tuple[int, ...]


@dataclass(frozen=True)
class DongEmbeddingResult:
    stego: np.ndarray
    auxiliary: DongAuxiliary
    embedded_bits: int
    changed_pixels: int


def difference_matrix(image: np.ndarray) -> np.ndarray:
    binary = as_binary(image)
    difference = np.empty_like(binary)
    difference[0, 0] = binary[0, 0]
    difference[1:, 0] = binary[1:, 0] ^ binary[:-1, 0]
    difference[:, 1:] = binary[:, 1:] ^ binary[:, :-1]
    return difference


def inverse_difference(difference: np.ndarray) -> np.ndarray:
    diff = as_binary(difference)
    image = np.empty_like(diff)
    image[0, 0] = diff[0, 0]
    for row in range(1, diff.shape[0]):
        image[row, 0] = diff[row, 0] ^ image[row - 1, 0]
    for row in range(diff.shape[0]):
        for col in range(1, diff.shape[1]):
            image[row, col] = diff[row, col] ^ image[row, col - 1]
    return image


def _pattern(stream: np.ndarray, head: int, stride: int) -> int:
    return int(
        (int(stream[head]) << 3)
        | (int(stream[head + stride]) << 2)
        | (int(stream[head + 2 * stride]) << 1)
        | int(stream[head + 3 * stride])
    )


def _write_pattern(
    stream: np.ndarray,
    head: int,
    stride: int,
    pattern: int,
) -> None:
    for offset, shift in enumerate((3, 2, 1, 0)):
        stream[head + offset * stride] = (pattern >> shift) & 1


def _triplet(counts: np.ndarray) -> tuple[int, int, int]:
    pm = min(range(1, 16), key=lambda pattern: (-int(counts[pattern]), pattern))
    pf = min(
        (pattern for pattern in SUBSTITUTABLE[pm] if pattern != 0),
        key=lambda pattern: (int(counts[pattern]), pattern),
    )
    parity = pf.bit_count() & 1
    pfr = min(
        (
            pattern
            for pattern in range(1, 16)
            if pattern not in (pm, pf) and (pattern.bit_count() & 1) == parity
        ),
        key=lambda pattern: (int(counts[pattern]), pattern),
    )
    return pm, pf, pfr


class _AdaptiveWindow:
    def __init__(
        self,
        stream: np.ndarray,
        context_length: int,
        position: int,
        stride: int,
        head_count: int,
    ) -> None:
        self.stream = stream
        self.context_length = max(1, context_length)
        self.stride = stride
        self.max_head = head_count - 1
        self.position = position
        self.start = position + 1
        self.end = min(self.max_head + 1, self.start + self.context_length)
        self.counts = np.zeros(16, dtype=np.int64)
        for head in range(self.start, self.end):
            self.counts[_pattern(stream, head, self.stride)] += 1

    def triplet(self) -> tuple[int, int, int]:
        return _triplet(self.counts)

    def replace(self, head: int, pattern: int) -> None:
        row = head % self.stride
        column = head // self.stride
        affected = [
            row + candidate_column * self.stride
            for candidate_column in range(max(0, column - 3), column + 4)
            if row + candidate_column * self.stride <= self.max_head
            and self.start <= row + candidate_column * self.stride < self.end
        ]
        old = [
            (candidate, _pattern(self.stream, candidate, self.stride))
            for candidate in affected
        ]
        _write_pattern(self.stream, head, self.stride, pattern)
        for candidate, old_pattern in old:
            new_pattern = _pattern(self.stream, candidate, self.stride)
            if new_pattern != old_pattern:
                self.counts[old_pattern] -= 1
                self.counts[new_pattern] += 1

    def move_forward(self) -> bool:
        if self.position >= self.max_head:
            return False
        new_position = self.position + 1
        new_start = new_position + 1
        new_end = min(self.max_head + 1, new_start + self.context_length)
        if self.start < self.end:
            self.counts[_pattern(self.stream, self.start, self.stride)] -= 1
        if new_end > self.end:
            self.counts[_pattern(self.stream, self.end, self.stride)] += 1
        self.position = new_position
        self.start = new_start
        self.end = new_end
        return True

    def move_backward(self) -> bool:
        if self.position <= 0:
            return False
        new_position = self.position - 1
        new_start = new_position + 1
        new_end = min(self.max_head + 1, new_start + self.context_length)
        if new_start < new_end:
            self.counts[_pattern(self.stream, new_start, self.stride)] += 1
        if new_end < self.end:
            self.counts[_pattern(self.stream, new_end, self.stride)] -= 1
        self.position = new_position
        self.start = new_start
        self.end = new_end
        return True


def _embed_excluding(
    stream: np.ndarray,
    payload: list[int],
    *,
    start: int,
    context_length: int,
    stride: int,
    head_count: int,
) -> tuple[int, list[int]]:
    if not payload:
        return start - 1, []
    data = deque(payload)
    location_map: list[int] = []
    window = _AdaptiveWindow(stream, context_length, start, stride, head_count)
    while True:
        head = window.position
        current = _pattern(stream, head, stride)
        pm, pf, pfr = window.triplet()
        if current == pm:
            bit = data.popleft()
            if bit:
                window.replace(head, pf)
            if not data:
                return head, location_map
        elif current == pf:
            window.replace(head, pfr)
            location_map.append(1)
        elif current == pfr:
            location_map.append(0)
        if not window.move_forward():
            raise ValueError("Insufficient Dong excluding-mode capacity")


def _extract_excluding(
    stream: np.ndarray,
    location_map: list[int],
    *,
    start: int,
    end: int,
    context_length: int,
    stride: int,
    head_count: int,
) -> list[int]:
    if end < start:
        return []
    marks = list(location_map)
    recovered: deque[int] = deque()
    window = _AdaptiveWindow(stream, context_length, end, stride, head_count)
    while True:
        head = window.position
        current = _pattern(stream, head, stride)
        pm, pf, pfr = window.triplet()
        if current == pm or current == pf:
            recovered.appendleft(0 if current == pm else 1)
            window.replace(head, pm)
        elif current == pfr:
            if not marks:
                raise ValueError("Dong location map is exhausted")
            mark = marks.pop()
            if mark:
                window.replace(head, pf)
        if head == start:
            break
        window.move_backward()
    if marks:
        raise ValueError("Dong location map contains unused marks")
    return list(recovered)


def serialize_auxiliary(auxiliary: DongAuxiliary) -> bytes:
    packed = np.packbits(
        np.asarray(auxiliary.location_map, dtype=np.uint8),
        bitorder="big",
    ).tobytes()
    compressed = zlib.compress(packed, level=9)
    raw = struct.pack(
        ">4sBII B i II",
        _MAGIC,
        _VERSION,
        auxiliary.image_shape[0],
        auxiliary.image_shape[1],
        auxiliary.context_divisor,
        auxiliary.end_position,
        auxiliary.secret_length,
        len(auxiliary.location_map),
    )
    return raw + compressed


def deserialize_auxiliary(payload: bytes) -> DongAuxiliary:
    header_format = ">4sBII B i II"
    header_size = struct.calcsize(header_format)
    values = struct.unpack_from(header_format, payload, 0)
    if values[0] != _MAGIC or values[1] != _VERSION:
        raise ValueError("Unsupported Dong auxiliary format")
    location_length = int(values[7])
    packed = zlib.decompress(payload[header_size:])
    location_map = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8),
        bitorder="big",
    )[:location_length]
    return DongAuxiliary(
        image_shape=(int(values[2]), int(values[3])),
        context_divisor=int(values[4]),
        end_position=int(values[5]),
        secret_length=int(values[6]),
        location_map=tuple(int(bit) for bit in location_map),
    )


def auxiliary_bits(auxiliary: DongAuxiliary) -> int:
    return 8 * len(serialize_auxiliary(auxiliary))


def embed(
    image: np.ndarray,
    message_bits: Iterable[int],
    *,
    context_divisor: int = 10,
) -> DongEmbeddingResult:
    original = as_binary(image)
    message = [int(bit) for bit in message_bits]
    if any(bit not in (0, 1) for bit in message):
        raise ValueError("Message bits must contain only zero and one")
    if not 1 <= context_divisor <= 10:
        raise ValueError("Dong context divisor must be in [1, 10]")
    difference = difference_matrix(original)
    stream = difference.reshape(-1, order="F").copy()
    stride = original.shape[0]
    head_count = original.shape[0] * max(original.shape[1] - 3, 0)
    context_length = max(1, round(original.size / context_divisor))
    if head_count == 0:
        raise ValueError("Dong embedding requires an image at least four pixels wide")

    end_position, location_map = _embed_excluding(
        stream,
        message,
        start=0,
        context_length=context_length,
        stride=stride,
        head_count=head_count,
    )
    stego = inverse_difference(stream.reshape(original.shape, order="F"))
    auxiliary = DongAuxiliary(
        image_shape=original.shape,
        context_divisor=context_divisor,
        end_position=end_position,
        secret_length=len(message),
        location_map=tuple(location_map),
    )
    return DongEmbeddingResult(
        stego=stego,
        auxiliary=auxiliary,
        embedded_bits=len(message),
        changed_pixels=int(np.count_nonzero(original != stego)),
    )


def extract(
    stego: np.ndarray,
    auxiliary: DongAuxiliary,
) -> tuple[np.ndarray, list[int]]:
    binary = as_binary(stego)
    if binary.shape != auxiliary.image_shape:
        raise ValueError("Stego shape does not match Dong auxiliary data")
    stream = difference_matrix(binary).reshape(-1, order="F").copy()
    stride = binary.shape[0]
    head_count = binary.shape[0] * max(binary.shape[1] - 3, 0)
    context_length = max(1, round(stream.size / auxiliary.context_divisor))
    message = _extract_excluding(
        stream,
        list(auxiliary.location_map),
        start=0,
        end=auxiliary.end_position,
        context_length=context_length,
        stride=stride,
        head_count=head_count,
    )
    if len(message) != auxiliary.secret_length:
        raise ValueError("Dong payload length does not match auxiliary data")
    restored = inverse_difference(stream.reshape(binary.shape, order="F"))
    return restored, message
