"""`.pq` binary file format used by the Mojo CLI port.

Layout (little-endian):
    bytes 0-1   : magic 'PQ' (0x50, 0x51)
    byte 2      : reserved (0)
    byte 3      : version (=1)
    bytes 4-7   : d (u32)
    bytes 8-15  : n (u64)
    byte 16     : bits (u8)
    bytes 17-31 : reserved (15 zero bytes)
    bytes 32+   : packed_indices (length = packed_nbytes(n*d, bits))
    then        : norms (n × float32, little-endian)

This is a minimal alternative to the Python `.npz` format that is trivially
parseable from Mojo without unzip/numpy-header machinery. See
`mojo/src/pq_format.mojo`.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from remex.packing import pack, unpack, packed_nbytes

if TYPE_CHECKING:
    from remex.core import CompressedVectors, Quantizer

PQ_HEADER_BYTES = 32
PQ_VERSION = 1
PQ_MAGIC = b"PQ\x00\x01"

PARAMS_HEADER_BYTES = 16
PARAMS_VERSION = 1
PARAMS_MAGIC = b"PR\x00\x01"


def save_params(path: str | Path, quantizer: "Quantizer") -> None:
    """Dump (R, boundaries, centroids) so a Mojo binary can mirror this Quantizer.

    Used to verify bit-identical encode output between Python and Mojo.
    """
    d = int(quantizer.d)
    bits = int(quantizer.bits)
    R = np.ascontiguousarray(quantizer.R, dtype=np.float32)
    boundaries = np.ascontiguousarray(quantizer.boundaries, dtype=np.float32)
    centroids = np.ascontiguousarray(quantizer.centroids, dtype=np.float32)
    if R.shape != (d, d):
        raise ValueError(f"R shape {R.shape} != (d, d) = ({d}, {d})")
    if boundaries.size != (1 << bits) - 1:
        raise ValueError("boundaries size mismatch")
    if centroids.size != (1 << bits):
        raise ValueError("centroids size mismatch")

    header = bytearray(PARAMS_HEADER_BYTES)
    header[0:4] = PARAMS_MAGIC
    header[4:8] = struct.pack("<I", d)
    header[8] = bits & 0xFF
    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(R.tobytes())
        f.write(boundaries.tobytes())
        f.write(centroids.tobytes())


def save_pq(path: str | Path, compressed: "CompressedVectors") -> None:
    """Serialize a CompressedVectors to the .pq binary format."""
    n, d, bits = int(compressed.n), int(compressed.d), int(compressed.bits)
    packed = pack(compressed.indices.ravel(), bits)
    expected_packed = packed_nbytes(n, d, bits)
    if packed.nbytes != expected_packed:
        raise ValueError(
            f"packed indices size mismatch: got {packed.nbytes}, "
            f"expected {expected_packed}"
        )

    header = bytearray(PQ_HEADER_BYTES)
    header[0:4] = PQ_MAGIC
    header[4:8] = struct.pack("<I", d)
    header[8:16] = struct.pack("<Q", n)
    header[16] = bits & 0xFF

    norms = np.ascontiguousarray(compressed.norms, dtype=np.float32)

    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(packed.tobytes())
        f.write(norms.tobytes())


def load_pq(path: str | Path) -> "CompressedVectors":
    """Read a .pq file and return a CompressedVectors."""
    from remex.core import CompressedVectors

    raw = Path(path).read_bytes()
    if len(raw) < PQ_HEADER_BYTES:
        raise ValueError(".pq file too small")
    if raw[0:2] != b"PQ":
        raise ValueError("bad .pq magic")
    if raw[3] != PQ_VERSION:
        raise ValueError(f"unsupported .pq version: {raw[3]}")

    (d,) = struct.unpack("<I", raw[4:8])
    (n,) = struct.unpack("<Q", raw[8:16])
    bits = raw[16]
    if bits in (5, 6, 7):
        raise ValueError(
            f"bits={bits} is not supported. Use 1-4 or 8 bits."
        )

    expected_packed = packed_nbytes(n, d, bits)
    expected_total = PQ_HEADER_BYTES + expected_packed + n * 4
    if len(raw) < expected_total:
        raise ValueError("truncated .pq data")

    packed = np.frombuffer(
        raw, dtype=np.uint8,
        count=expected_packed,
        offset=PQ_HEADER_BYTES,
    )
    norms = np.frombuffer(
        raw, dtype=np.float32,
        count=n,
        offset=PQ_HEADER_BYTES + expected_packed,
    )

    indices = unpack(packed, bits, n * d).reshape(n, d)
    return CompressedVectors(indices.copy(), norms.copy(), d, bits)
