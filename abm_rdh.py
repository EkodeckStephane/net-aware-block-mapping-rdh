"""Reproducible adaptive block-mapping RDH for binary images."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, floor, log2, sqrt
import struct
from time import perf_counter
from typing import Any, Iterable
import zlib

import numpy as np


BLOCK_SIZE = 3
PATTERN_COUNT = 2 ** (BLOCK_SIZE * BLOCK_SIZE)
UNIFORM_PATTERNS = {0, PATTERN_COUNT - 1}


@dataclass(frozen=True)
class AuxiliaryData:
    """Side information required for exact extraction and restoration."""

    mapping_tables: dict[int, dict[int, int]]
    payload_length: int
    image_shape: tuple[int, int]
    block_size: int = BLOCK_SIZE
    alpha: float = 0.1
    policy: str = "adaptive"


@dataclass(frozen=True)
class EmbeddingResult:
    stego: np.ndarray
    auxiliary: AuxiliaryData
    embedded_bits: int
    modified_blocks: int


_AUXILIARY_MAGIC = b"ABMA"
_POLICY_CODES = {
    "adaptive": 0,
    "hamming1": 1,
    "cnn": 2,
    "cnn_hamming1": 3,
}
_POLICY_NAMES = {code: name for name, code in _POLICY_CODES.items()}


def _wire_encode(raw: bytes) -> bytes:
    compressed = zlib.compress(raw, level=9)
    if len(compressed) + 1 < len(raw) + 1:
        return b"\x01" + compressed
    return b"\x00" + raw


def _wire_decode(payload: bytes) -> bytes:
    if not payload:
        raise ValueError("Auxiliary payload is empty")
    if payload[0] == 0:
        return payload[1:]
    if payload[0] == 1:
        return zlib.decompress(payload[1:])
    raise ValueError("Unknown auxiliary compression marker")


def _pack_fixed_width(values: list[int], width: int) -> bytes:
    accumulator = 0
    bit_count = 0
    output = bytearray()
    for value in values:
        accumulator = (accumulator << width) | value
        bit_count += width
        while bit_count >= 8:
            bit_count -= 8
            output.append((accumulator >> bit_count) & 0xFF)
    if bit_count:
        output.append((accumulator << (8 - bit_count)) & 0xFF)
    return bytes(output)


def _unpack_fixed_width(payload: bytes, count: int, width: int) -> list[int]:
    accumulator = 0
    bit_count = 0
    output = []
    for byte in payload:
        accumulator = (accumulator << 8) | byte
        bit_count += 8
        while bit_count >= width and len(output) < count:
            bit_count -= width
            output.append((accumulator >> bit_count) & ((1 << width) - 1))
    if len(output) != count:
        raise ValueError("Truncated fixed-width auxiliary payload")
    return output


def _combination_rank(values: list[int]) -> int:
    """Rank a sorted subset using the combinatorial number system."""

    return sum(comb(value, index) for index, value in enumerate(values, start=1))


def _combination_unrank(rank: int, count: int) -> list[int]:
    values = [0] * count
    upper = PATTERN_COUNT - 1
    for index in range(count, 0, -1):
        value = upper
        while comb(value, index) > rank:
            value -= 1
        values[index - 1] = value
        rank -= comb(value, index)
        upper = value - 1
    if rank != 0:
        raise ValueError("Corrupt enumerative Hamming-1 peak set")
    return values


def serialize_auxiliary(auxiliary: AuxiliaryData) -> bytes:
    """Serialize all base-layer side information into a compact wire format."""

    if auxiliary.policy not in _POLICY_CODES:
        raise ValueError(f"Unsupported auxiliary policy: {auxiliary.policy}")
    if auxiliary.policy in ("hamming1", "cnn_hamming1") and all(
        len(table) == 2 and table.get(0) == peak
        for peak, table in auxiliary.mapping_tables.items()
    ):
        peaks = sorted(auxiliary.mapping_tables)
        shifts = []
        for peak in peaks:
            zero = auxiliary.mapping_tables[peak][1]
            difference = peak ^ zero
            if difference.bit_count() != 1:
                raise ValueError("Hamming-1 auxiliary contains a non-unit mapping")
            shifts.append(difference.bit_length() - 1)
        shift_value = 0
        for shift in shifts:
            shift_value = shift_value * 9 + shift
        combined = _combination_rank(peaks) * (9 ** len(peaks)) + shift_value
        state_count = comb(PATTERN_COUNT, len(peaks)) * (9 ** len(peaks))
        body_size = max(0, (max(state_count - 1, 0).bit_length() + 7) // 8)
        body = combined.to_bytes(body_size, "big") if body_size else b""
        raw = struct.pack(
            ">4sBIH",
            _AUXILIARY_MAGIC,
            4,
            int(auxiliary.payload_length),
            len(peaks),
        )
        return b"\x00" + raw + body
    raw = bytearray(
        struct.pack(
            ">4sBIII BfH",
            _AUXILIARY_MAGIC,
            1,
            int(auxiliary.image_shape[0]),
            int(auxiliary.image_shape[1]),
            int(auxiliary.payload_length),
            int(auxiliary.block_size),
            float(auxiliary.alpha),
            len(auxiliary.mapping_tables),
        )
    )
    raw.append(_POLICY_CODES[auxiliary.policy])
    for peak in sorted(auxiliary.mapping_tables):
        table = auxiliary.mapping_tables[peak]
        states = [table[symbol] for symbol in sorted(table)]
        raw.extend(struct.pack(">HH", int(peak), len(states)))
        raw.extend(struct.pack(f">{len(states)}H", *states))
    return _wire_encode(bytes(raw))


def deserialize_auxiliary(
    payload: bytes,
    *,
    image_shape: tuple[int, int] | None = None,
) -> AuxiliaryData:
    """Reconstruct base-layer side information from ``serialize_auxiliary``."""

    raw = _wire_decode(payload)
    magic, version = struct.unpack_from(">4sB", raw, 0)
    if magic != _AUXILIARY_MAGIC:
        raise ValueError("Unsupported base auxiliary format")
    if version == 4:
        header_format = ">4sBIH"
        header_size = struct.calcsize(header_format)
        _, _, payload_length, table_count = struct.unpack_from(
            header_format, raw, 0
        )
        combined = int.from_bytes(raw[header_size:], "big")
        shift_modulus = 9 ** table_count
        peak_rank, shift_value = divmod(combined, shift_modulus)
        peaks = _combination_unrank(peak_rank, table_count)
        shifts = [0] * table_count
        for index in range(table_count - 1, -1, -1):
            shift_value, shifts[index] = divmod(shift_value, 9)
        if shift_value:
            raise ValueError("Corrupt enumerative Hamming-1 shift stream")
        mapping_tables = {
            peak: {0: peak, 1: peak ^ (1 << shift)}
            for peak, shift in zip(peaks, shifts)
        }
        return AuxiliaryData(
            mapping_tables=mapping_tables,
            payload_length=int(payload_length),
            image_shape=image_shape or (0, 0),
            block_size=BLOCK_SIZE,
            alpha=0.0,
            policy="hamming1",
        )
    if version == 3:
        header_format = ">4sBIHB"
        header_size = struct.calcsize(header_format)
        _, _, payload_length, table_count, mode = struct.unpack_from(
            header_format, raw, 0
        )
        body = raw[header_size:]
        mapping_tables = {}
        if mode == 0:
            bitmap = body[:64]
            packed = body[64:]
            peaks = [
                peak
                for peak in range(PATTERN_COUNT)
                if bitmap[peak // 8] & (1 << (peak % 8))
            ]
            if len(peaks) != table_count:
                raise ValueError("Corrupt dense Hamming-1 peak bitmap")
            for index, peak in enumerate(peaks):
                byte = packed[index // 2]
                shift = (byte & 0x0F) if index % 2 == 0 else (byte >> 4)
                mapping_tables[peak] = {0: peak, 1: peak ^ (1 << shift)}
        elif mode == 1:
            values = _unpack_fixed_width(body, table_count, 13)
            for value in values:
                peak = value >> 4
                shift = value & 0x0F
                mapping_tables[peak] = {0: peak, 1: peak ^ (1 << shift)}
        else:
            raise ValueError("Unknown compact Hamming-1 auxiliary mode")
        return AuxiliaryData(
            mapping_tables=mapping_tables,
            payload_length=int(payload_length),
            image_shape=image_shape or (0, 0),
            block_size=BLOCK_SIZE,
            alpha=0.0,
            policy="hamming1",
        )
    if version == 2:
        header_format = ">4sBIII BfBH"
        header_size = struct.calcsize(header_format)
        (
            _,
            _,
            height,
            width,
            payload_length,
            block_size,
            alpha,
            policy_code,
            table_count,
        ) = struct.unpack_from(header_format, raw, 0)
        bitmap = raw[header_size : header_size + 64]
        packed = raw[header_size + 64 :]
        peaks = [
            peak
            for peak in range(PATTERN_COUNT)
            if bitmap[peak // 8] & (1 << (peak % 8))
        ]
        if len(peaks) != table_count:
            raise ValueError("Corrupt Hamming-1 peak bitmap")
        mapping_tables = {}
        for index, peak in enumerate(peaks):
            byte = packed[index // 2]
            shift = (byte & 0x0F) if index % 2 == 0 else (byte >> 4)
            mapping_tables[peak] = {0: peak, 1: peak ^ (1 << shift)}
        return AuxiliaryData(
            mapping_tables=mapping_tables,
            payload_length=int(payload_length),
            image_shape=(int(height), int(width)),
            block_size=int(block_size),
            alpha=float(alpha),
            policy=_POLICY_NAMES[int(policy_code)],
        )
    header_format = ">4sBIII BfH"
    header_size = struct.calcsize(header_format)
    (
        magic,
        version,
        height,
        width,
        payload_length,
        block_size,
        alpha,
        table_count,
    ) = struct.unpack_from(header_format, raw, 0)
    if version != 1:
        raise ValueError("Unsupported base auxiliary format")
    offset = header_size
    policy_code = raw[offset]
    offset += 1
    mapping_tables: dict[int, dict[int, int]] = {}
    for _ in range(table_count):
        peak, state_count = struct.unpack_from(">HH", raw, offset)
        offset += 4
        states = struct.unpack_from(f">{state_count}H", raw, offset)
        offset += 2 * state_count
        mapping_tables[int(peak)] = {
            symbol: int(pattern) for symbol, pattern in enumerate(states)
        }
    return AuxiliaryData(
        mapping_tables=mapping_tables,
        payload_length=int(payload_length),
        image_shape=(int(height), int(width)),
        block_size=int(block_size),
        alpha=float(alpha),
        policy=_POLICY_NAMES[int(policy_code)],
    )


def auxiliary_bits(auxiliary: AuxiliaryData) -> int:
    return 8 * len(serialize_auxiliary(auxiliary))


def as_binary(image: np.ndarray) -> np.ndarray:
    """Return a two-dimensional uint8 image containing only zero and one."""

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("Binary RDH expects a two-dimensional image")
    return (array > 0).astype(np.uint8)


def iter_blocks(image: np.ndarray) -> Iterable[tuple[int, int, np.ndarray]]:
    """Yield complete non-overlapping 3x3 blocks in raster order."""

    height, width = image.shape
    usable_height = height - height % BLOCK_SIZE
    usable_width = width - width % BLOCK_SIZE
    for row in range(0, usable_height, BLOCK_SIZE):
        for col in range(0, usable_width, BLOCK_SIZE):
            yield row, col, image[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE]


def pattern_index(block: np.ndarray) -> int:
    bits = as_binary(block).reshape(-1)
    if bits.size != BLOCK_SIZE * BLOCK_SIZE:
        raise ValueError("A pattern must contain exactly nine pixels")
    weights = 1 << np.arange(bits.size - 1, -1, -1)
    return int(bits @ weights)


def pattern_array(pattern: int) -> np.ndarray:
    if not 0 <= pattern < PATTERN_COUNT:
        raise ValueError(f"Pattern must be in [0, {PATTERN_COUNT - 1}]")
    bits = np.array(
        [(pattern >> shift) & 1 for shift in range(8, -1, -1)],
        dtype=np.uint8,
    )
    return bits.reshape(BLOCK_SIZE, BLOCK_SIZE)


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _histogram(image: np.ndarray) -> tuple[list[int], dict[int, list[tuple[int, int]]]]:
    raster_patterns = []
    positions: dict[int, list[tuple[int, int]]] = {}
    for row, col, block in iter_blocks(image):
        pattern = pattern_index(block)
        raster_patterns.append(pattern)
        positions.setdefault(pattern, []).append((row, col))
    return raster_patterns, positions


def build_mapping_tables(
    image: np.ndarray,
    *,
    alpha: float = 0.1,
    min_frequency: int = 2,
    exclude_uniform: bool = True,
    policy: str = "adaptive",
    candidate_ranker: Any | None = None,
) -> dict[int, dict[int, int]]:
    """Build disjoint adaptive mapping tables.

    Symbol zero preserves the PEAK pattern. Remaining symbols map to disjoint
    ZERO patterns. The ``adaptive`` policy uses H_T = mean + alpha * std.
    The ``hamming1`` policy permits one ZERO at Hamming distance exactly one.
    """

    binary = as_binary(image)
    _, positions = _histogram(binary)
    peaks = [
        pattern
        for pattern, occurrences in positions.items()
        if len(occurrences) >= min_frequency
        and (not exclude_uniform or pattern not in UNIFORM_PATTERNS)
    ]
    peaks.sort(key=lambda pattern: (-len(positions[pattern]), pattern))

    zeros = [pattern for pattern in range(PATTERN_COUNT) if pattern not in positions]
    used_zeros: set[int] = set()
    tables: dict[int, dict[int, int]] = {}

    for peak in peaks:
        available = [pattern for pattern in zeros if pattern not in used_zeros]
        if not available:
            break

        ranked = sorted(
            ((hamming(peak, pattern), pattern) for pattern in available),
            key=lambda item: (item[0], item[1]),
        )
        if policy in ("adaptive", "cnn"):
            distances = np.fromiter((distance for distance, _ in ranked), dtype=float)
            threshold = float(distances.mean() + alpha * distances.std())
            candidates = [
                pattern for distance, pattern in ranked if distance <= threshold
            ]
            if policy == "cnn":
                if candidate_ranker is None:
                    raise ValueError("The cnn policy requires a candidate ranker")
                predicted_costs = candidate_ranker.predict_costs(
                    binary,
                    peak,
                    candidates,
                )
                candidates = [
                    candidate
                    for _, candidate in sorted(
                        zip(predicted_costs, candidates),
                        key=lambda item: (float(item[0]), item[1]),
                    )
                ]
        elif policy in ("hamming1", "cnn_hamming1"):
            candidates = [pattern for distance, pattern in ranked if distance == 1]
            if policy == "cnn_hamming1" and candidates:
                if candidate_ranker is None:
                    raise ValueError(
                        "The cnn_hamming1 policy requires a candidate ranker"
                    )
                predicted_costs = candidate_ranker.predict_costs(
                    binary,
                    peak,
                    candidates,
                )
                candidates = [
                    candidate
                    for _, candidate in sorted(
                        zip(predicted_costs, candidates),
                        key=lambda item: (float(item[0]), item[1]),
                    )
                ]
        else:
            raise ValueError(f"Unknown mapping policy: {policy}")

        state_count = 1 << floor(log2(len(candidates) + 1))
        if policy in ("hamming1", "cnn_hamming1"):
            state_count = min(state_count, 2)
        if state_count < 2:
            continue

        selected = candidates[: state_count - 1]
        table = {0: peak}
        table.update({symbol: pattern for symbol, pattern in enumerate(selected, start=1)})
        tables[peak] = table
        used_zeros.update(selected)

    return tables


def capacity_bits(image: np.ndarray, mapping_tables: dict[int, dict[int, int]]) -> int:
    binary = as_binary(image)
    return sum(
        int(log2(len(mapping_tables[pattern])))
        for _, _, block in iter_blocks(binary)
        if (pattern := pattern_index(block)) in mapping_tables
    )


def optimize_tables_for_net_capacity(
    image: np.ndarray,
    mapping_tables: dict[int, dict[int, int]],
    *,
    policy: str,
    alpha: float = 0.1,
) -> dict[int, dict[int, int]]:
    """Keep the table subset maximizing payload minus serialized side data."""

    binary = as_binary(image)
    _, positions = _histogram(binary)
    ordered = sorted(
        mapping_tables,
        key=lambda peak: (-len(positions.get(peak, ())), peak),
    )
    best_net = 0
    best_count = 0
    current: dict[int, dict[int, int]] = {}
    gross = 0
    for count, peak in enumerate(ordered, start=1):
        current[peak] = mapping_tables[peak]
        gross += len(positions.get(peak, ())) * int(
            log2(len(mapping_tables[peak]))
        )
        auxiliary = AuxiliaryData(
            mapping_tables=dict(current),
            payload_length=gross,
            image_shape=binary.shape,
            alpha=alpha,
            policy=policy,
        )
        net = gross - auxiliary_bits(auxiliary)
        if net > best_net:
            best_net = net
            best_count = count
    return {peak: mapping_tables[peak] for peak in ordered[:best_count]}


def embed(
    image: np.ndarray,
    message_bits: Iterable[int],
    *,
    alpha: float = 0.1,
    min_frequency: int = 2,
    policy: str = "adaptive",
    candidate_ranker: Any | None = None,
    mapping_tables: dict[int, dict[int, int]] | None = None,
) -> EmbeddingResult:
    """Embed a bitstream in raster order and return explicit side information."""

    original = as_binary(image)
    message = [int(bit) for bit in message_bits]
    if any(bit not in (0, 1) for bit in message):
        raise ValueError("Message bits must contain only zero and one")

    tables = mapping_tables
    if tables is None:
        tables = build_mapping_tables(
            original,
            alpha=alpha,
            min_frequency=min_frequency,
            policy=policy,
            candidate_ranker=candidate_ranker,
        )
    maximum = capacity_bits(original, tables)
    if len(message) > maximum:
        raise ValueError(f"Payload has {len(message)} bits but capacity is {maximum}")

    stego = original.copy()
    bit_offset = 0
    modified_blocks = 0

    for row, col, block in iter_blocks(original):
        peak = pattern_index(block)
        table = tables.get(peak)
        if table is None or bit_offset >= len(message):
            continue

        bits_per_block = int(log2(len(table)))
        remaining = min(bits_per_block, len(message) - bit_offset)
        chunk = message[bit_offset : bit_offset + remaining]
        chunk.extend([0] * (bits_per_block - remaining))
        symbol = 0
        for bit in chunk:
            symbol = (symbol << 1) | bit
        replacement = table[symbol]
        stego[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE] = pattern_array(
            replacement
        )
        modified_blocks += replacement != peak
        bit_offset += remaining

    auxiliary = AuxiliaryData(
        mapping_tables=tables,
        payload_length=bit_offset,
        image_shape=original.shape,
        alpha=alpha,
        policy=policy,
    )
    return EmbeddingResult(stego, auxiliary, bit_offset, modified_blocks)


def extract(stego: np.ndarray, auxiliary: AuxiliaryData) -> tuple[np.ndarray, list[int]]:
    """Extract the exact payload and restore the original image."""

    binary = as_binary(stego)
    if auxiliary.image_shape != (0, 0) and binary.shape != auxiliary.image_shape:
        raise ValueError("Stego image shape does not match auxiliary data")

    inverse: dict[int, tuple[int, int, int]] = {}
    for peak, table in auxiliary.mapping_tables.items():
        bits_per_block = int(log2(len(table)))
        for symbol, pattern in table.items():
            if pattern in inverse:
                raise ValueError("Mapping tables are not globally injective")
            inverse[pattern] = (peak, symbol, bits_per_block)

    restored = binary.copy()
    recovered: list[int] = []
    for row, col, block in iter_blocks(binary):
        if len(recovered) >= auxiliary.payload_length:
            break
        pattern = pattern_index(block)
        decoded = inverse.get(pattern)
        if decoded is None:
            continue

        peak, symbol, bits_per_block = decoded
        bits = [
            (symbol >> shift) & 1
            for shift in range(bits_per_block - 1, -1, -1)
        ]
        remaining = auxiliary.payload_length - len(recovered)
        recovered.extend(bits[:remaining])
        restored[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE] = pattern_array(peak)

    if len(recovered) != auxiliary.payload_length:
        raise ValueError("Stego image does not contain the declared payload")
    return restored, recovered


def psnr(original: np.ndarray, stego: np.ndarray) -> float:
    """PSNR for binary images represented on the normalized [0, 1] scale."""

    left = as_binary(original).astype(np.float64)
    right = as_binary(stego).astype(np.float64)
    if left.shape != right.shape:
        raise ValueError("PSNR inputs must have identical shapes")
    mse = float(np.mean((left - right) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def _drd_weights() -> np.ndarray:
    weights = np.zeros((5, 5), dtype=np.float64)
    for row in range(5):
        for col in range(5):
            if row == 2 and col == 2:
                continue
            weights[row, col] = 1.0 / sqrt((row - 2) ** 2 + (col - 2) ** 2)
    return weights / weights.sum()


DRD_WEIGHTS = _drd_weights()


def drd(original: np.ndarray, stego: np.ndarray) -> float:
    """Compute the standard distance-reciprocal distortion measure."""

    cover = as_binary(original)
    marked = as_binary(stego)
    if cover.shape != marked.shape:
        raise ValueError("DRD inputs must have identical shapes")

    changed_rows, changed_cols = np.where(cover != marked)
    if changed_rows.size == 0:
        return 0.0

    height, width = cover.shape
    distortion = 0.0
    for row, col in zip(changed_rows, changed_cols):
        flipped_value = marked[row, col]
        for drow in range(-2, 3):
            for dcol in range(-2, 3):
                source_row = row + drow
                source_col = col + dcol
                if 0 <= source_row < height and 0 <= source_col < width:
                    distortion += (
                        abs(int(cover[source_row, source_col]) - int(flipped_value))
                        * DRD_WEIGHTS[drow + 2, dcol + 2]
                    )

    non_uniform_blocks = 0
    for row in range(0, height - height % 8, 8):
        for col in range(0, width - width % 8, 8):
            block = cover[row : row + 8, col : col + 8]
            non_uniform_blocks += bool(block.min() != block.max())
    return distortion / max(non_uniform_blocks, 1)


def evaluate(
    image: np.ndarray,
    message_bits: Iterable[int],
    *,
    alpha: float = 0.1,
    policy: str = "adaptive",
    candidate_ranker: Any | None = None,
) -> tuple[EmbeddingResult, np.ndarray, list[int], dict[str, float | int | bool]]:
    message = list(message_bits)
    start = perf_counter()
    embedded = embed(
        image,
        message,
        alpha=alpha,
        policy=policy,
        candidate_ranker=candidate_ranker,
    )
    after_embed = perf_counter()
    restored, recovered = extract(embedded.stego, embedded.auxiliary)
    after_extract = perf_counter()

    original = as_binary(image)
    metrics: dict[str, float | int | bool] = {
        "capacity_bits": embedded.embedded_bits,
        "modified_blocks": embedded.modified_blocks,
        "changed_pixels": int(np.count_nonzero(original != embedded.stego)),
        "psnr_db": psnr(original, embedded.stego),
        "drd": drd(original, embedded.stego),
        "embedding_seconds": after_embed - start,
        "extraction_seconds": after_extract - after_embed,
        "reversible": bool(np.array_equal(original, restored)),
        "message_exact": message == recovered,
    }
    return embedded, restored, recovered, metrics
