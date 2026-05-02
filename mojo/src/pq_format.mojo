"""`.pq` file format: simple binary container for compressed vectors.

Layout (all little-endian):
  bytes 0-1   : magic 'PQ' (0x50, 0x51)
  byte 2      : reserved (0)
  byte 3      : version (=1)
  bytes 4-7   : d (u32)
  bytes 8-15  : n (u64)
  byte 16     : bits (u8)
  bytes 17-31 : reserved (15 zero bytes) — header padded to 32 bytes
  bytes 32+   : packed_indices (length = packed_nbytes(n*d, bits))
  then        : norms (n × float32)

The Python side mirrors this in `remex.pq_format.{save_pq, load_pq}`.
"""

from std.pathlib import Path
from std.memory import alloc, UnsafePointer
from src.packing import packed_nbytes


comptime PQ_HEADER_BYTES = 32
comptime PQ_VERSION: UInt8 = 1


struct PqVectors(Movable):
    """In-memory container for a loaded .pq file."""
    var packed_indices: UnsafePointer[UInt8, MutExternalOrigin]
    var norms: UnsafePointer[Float32, MutExternalOrigin]
    var n: Int
    var d: Int
    var bits: Int
    var packed_bytes: Int

    def __init__(out self, n: Int, d: Int, bits: Int) raises:
        self.n = n
        self.d = d
        self.bits = bits
        self.packed_bytes = packed_nbytes(n * d, bits)
        self.packed_indices = alloc[UInt8](self.packed_bytes)
        self.norms = alloc[Float32](n)

    def __del__(deinit self):
        self.packed_indices.free()
        self.norms.free()


def _u32_le(buf: UnsafePointer[UInt8, MutExternalOrigin], off: Int) -> Int:
    return (
        Int(buf[off])
        | (Int(buf[off + 1]) << 8)
        | (Int(buf[off + 2]) << 16)
        | (Int(buf[off + 3]) << 24)
    )


def _u64_le(buf: UnsafePointer[UInt8, MutExternalOrigin], off: Int) -> Int:
    var v: Int = 0
    for k in range(8):
        v |= Int(buf[off + k]) << (8 * k)
    return v


def save_pq(path: String,
            packed_indices: UnsafePointer[UInt8, MutExternalOrigin], packed_bytes: Int,
            norms: UnsafePointer[Float32, MutExternalOrigin], n: Int,
            d: Int, bits: Int) raises:
    """Write a .pq file."""
    var total = PQ_HEADER_BYTES + packed_bytes + n * 4
    var buf = alloc[UInt8](total)
    # Zero header
    for i in range(PQ_HEADER_BYTES):
        buf[i] = UInt8(0)
    buf[0] = UInt8(0x50)  # 'P'
    buf[1] = UInt8(0x51)  # 'Q'
    buf[2] = UInt8(0)
    buf[3] = PQ_VERSION
    # d (u32 LE) at offset 4
    buf[4] = UInt8(d & 0xFF)
    buf[5] = UInt8((d >> 8) & 0xFF)
    buf[6] = UInt8((d >> 16) & 0xFF)
    buf[7] = UInt8((d >> 24) & 0xFF)
    # n (u64 LE) at offset 8
    for k in range(8):
        buf[8 + k] = UInt8((n >> (8 * k)) & 0xFF)
    # bits at offset 16
    buf[16] = UInt8(bits)

    # packed indices
    for i in range(packed_bytes):
        buf[PQ_HEADER_BYTES + i] = packed_indices[i]

    # norms (float32 LE)
    var norms_bytes = norms.bitcast[UInt8]()
    var norm_off = PQ_HEADER_BYTES + packed_bytes
    for i in range(n * 4):
        buf[norm_off + i] = norms_bytes[i]

    # Convert to List[UInt8] and write via Path.write_bytes
    var span = Span[UInt8, MutExternalOrigin](ptr=buf, length=total)
    Path(path).write_bytes(span)
    buf.free()


def load_pq(path: String) raises -> PqVectors:
    """Read a .pq file."""
    var raw = Path(path).read_bytes()
    if len(raw) < PQ_HEADER_BYTES:
        raise Error(".pq file too small")
    if Int(raw[0]) != 0x50 or Int(raw[1]) != 0x51:
        raise Error("bad .pq magic")
    if Int(raw[3]) != Int(PQ_VERSION):
        raise Error("unsupported .pq version")

    # Copy the raw bytes into an owned buffer for pointer arithmetic.
    var owned = alloc[UInt8](len(raw))
    for i in range(len(raw)):
        owned[i] = raw[i]

    var d = _u32_le(owned, 4)
    var n = _u64_le(owned, 8)
    var bits = Int(owned[16])

    var pq = PqVectors(n, d, bits)
    var expected = PQ_HEADER_BYTES + pq.packed_bytes + n * 4
    if len(raw) < expected:
        owned.free()
        raise Error("truncated .pq data")

    for i in range(pq.packed_bytes):
        pq.packed_indices[i] = owned[PQ_HEADER_BYTES + i]
    var norms_bytes = pq.norms.bitcast[UInt8]()
    var norm_off = PQ_HEADER_BYTES + pq.packed_bytes
    for i in range(n * 4):
        norms_bytes[i] = owned[norm_off + i]

    owned.free()
    return pq^
