"""`.params` file: dump of (R, boundaries, centroids) so a Mojo binary can
encode / search using the *exact* parameters a Python `remex.Quantizer`
would use. Lets us write tests that assert bit-identical encode output.

Layout (little-endian):
  bytes 0-3   : magic 'PR\x00\x01'
  bytes 4-7   : d (u32)
  byte 8      : bits (u8)
  bytes 9-15  : reserved (zero)
  bytes 16+   : R              (d * d float32 row-major)
  then        : boundaries     ((n_levels - 1) float32)
  then        : centroids      (n_levels float32)
"""

from std.pathlib import Path
from std.memory import alloc, UnsafePointer
from src.codebook import Codebook
from src.matrix import Matrix


comptime PARAMS_HEADER_BYTES = 16
comptime PARAMS_VERSION: UInt8 = 1


def _u32_le(buf: UnsafePointer[UInt8, MutExternalOrigin], off: Int) -> Int:
    return (
        Int(buf[off])
        | (Int(buf[off + 1]) << 8)
        | (Int(buf[off + 2]) << 16)
        | (Int(buf[off + 3]) << 24)
    )


def load_params(path: String, mut R_out: Matrix, mut cb_out: Codebook) raises:
    """Read R, boundaries, centroids from a .params file into the given outputs.

    `R_out` must be (d, d) and `cb_out` must have matching `bits` already.
    """
    var raw = Path(path).read_bytes()
    if len(raw) < PARAMS_HEADER_BYTES:
        raise Error(".params file too small")

    var owned = alloc[UInt8](len(raw))
    for i in range(len(raw)):
        owned[i] = raw[i]

    if Int(owned[0]) != Int(ord("P")) or Int(owned[1]) != Int(ord("R")):
        owned.free()
        raise Error("bad .params magic")
    if Int(owned[3]) != Int(PARAMS_VERSION):
        owned.free()
        raise Error("unsupported .params version")

    var d = _u32_le(owned, 4)
    var bits = Int(owned[8])
    if d != R_out.rows or d != R_out.cols:
        owned.free()
        raise Error("R shape mismatch")
    if bits != cb_out.bits:
        owned.free()
        raise Error("bits mismatch")

    var n_levels = 1 << bits
    var R_bytes = d * d * 4
    var b_bytes = (n_levels - 1) * 4
    var c_bytes = n_levels * 4
    var expected = PARAMS_HEADER_BYTES + R_bytes + b_bytes + c_bytes
    if len(raw) < expected:
        owned.free()
        raise Error("truncated .params data")

    # Copy R
    var R_dst = R_out.data.bitcast[UInt8]()
    var R_off = PARAMS_HEADER_BYTES
    for i in range(R_bytes):
        R_dst[i] = owned[R_off + i]

    var b_dst = cb_out.boundaries.bitcast[UInt8]()
    var b_off = R_off + R_bytes
    for i in range(b_bytes):
        b_dst[i] = owned[b_off + i]

    var c_dst = cb_out.centroids.bitcast[UInt8]()
    var c_off = b_off + b_bytes
    for i in range(c_bytes):
        c_dst[i] = owned[c_off + i]

    owned.free()


def save_params(path: String, R: Matrix, cb: Codebook) raises:
    """Write (R, boundaries, centroids) to a .params file."""
    var d = R.rows
    var bits = cb.bits
    var n_levels = cb.n_levels
    var R_bytes = d * d * 4
    var b_bytes = (n_levels - 1) * 4
    var c_bytes = n_levels * 4
    var total = PARAMS_HEADER_BYTES + R_bytes + b_bytes + c_bytes

    var buf = alloc[UInt8](total)
    for i in range(PARAMS_HEADER_BYTES):
        buf[i] = UInt8(0)
    buf[0] = UInt8(0x50)  # 'P'
    buf[1] = UInt8(0x52)  # 'R'
    buf[3] = PARAMS_VERSION
    buf[4] = UInt8(d & 0xFF)
    buf[5] = UInt8((d >> 8) & 0xFF)
    buf[6] = UInt8((d >> 16) & 0xFF)
    buf[7] = UInt8((d >> 24) & 0xFF)
    buf[8] = UInt8(bits)

    var R_src = R.data.bitcast[UInt8]()
    for i in range(R_bytes):
        buf[PARAMS_HEADER_BYTES + i] = R_src[i]
    var b_src = cb.boundaries.bitcast[UInt8]()
    for i in range(b_bytes):
        buf[PARAMS_HEADER_BYTES + R_bytes + i] = b_src[i]
    var c_src = cb.centroids.bitcast[UInt8]()
    for i in range(c_bytes):
        buf[PARAMS_HEADER_BYTES + R_bytes + b_bytes + i] = c_src[i]

    var span = Span[UInt8, MutExternalOrigin](ptr=buf, length=total)
    Path(path).write_bytes(span)
    buf.free()
