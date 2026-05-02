"""Memory-efficient packed storage for quantized vectors.

Mirrors `remex.core.PackedVectors`: indices are stored bit-packed in a
single contiguous `(n, row_bytes)` buffer with row-aligned offsets, and
unpacked on demand. For sub-byte widths this uses 2-4x less RAM than the
unpacked uint8 layout used by `CompressedVectors`.

`row_aligned` is true when `(d * bits) % 8 == 0`. In that case the
flat-packed and row-by-row-packed layouts coincide, so a single
`pack(flat, n*d, bits, ...)` produces the same bytes as packing each
row independently. When false (e.g. d=10 at 3 bits), each row carries
internal padding and must be packed/unpacked individually.

The on-disk `.pq` format always stores indices flat-packed across the
whole `n*d` value stream — which is identical to the row-aligned
PackedVectors layout, but not the row-padded one. `from_pq` and
`to_pq_bytes` handle that distinction.
"""

from std.memory import alloc, UnsafePointer
from src.packing import pack, unpack, packed_nbytes


struct PackedVectors(Movable):
    """Bit-packed (n, row_bytes) index storage with on-demand unpack.

    Owning struct: `__init__` allocates `packed` and `norms`, `__del__`
    frees them. Use `from_indices` to build from an unpacked (n, d) uint8
    buffer; use `from_pq_bytes` to adopt a flat `.pq`-style packed buffer.
    """
    var packed: UnsafePointer[UInt8, MutExternalOrigin]      # (n * row_bytes)
    var norms: UnsafePointer[Float32, MutExternalOrigin]     # (n,)
    var n: Int
    var d: Int
    var bits: Int
    var row_bytes: Int
    var row_aligned: Bool

    def __init__(out self, n: Int, d: Int, bits: Int) raises:
        self.n = n
        self.d = d
        self.bits = bits
        self.row_bytes = packed_nbytes(d, bits)
        self.row_aligned = ((d * bits) % 8) == 0
        self.packed = alloc[UInt8](n * self.row_bytes)
        self.norms = alloc[Float32](n)

    def __del__(deinit self):
        self.packed.free()
        self.norms.free()


def from_indices(indices: UnsafePointer[UInt8, MutExternalOrigin],
                 norms: UnsafePointer[Float32, MutExternalOrigin],
                 n: Int, d: Int, bits: Int) raises -> PackedVectors:
    """Build a PackedVectors from an unpacked (n, d) uint8 buffer."""
    var pv = PackedVectors(n, d, bits)
    var packed_buf = pv.packed
    if pv.row_aligned:
        # Flat layout matches: pack the whole stream in one shot.
        pack(indices, n * d, bits, packed_buf)
    else:
        # Pack each row independently into its own row_bytes-sized slot.
        # `pack` requires a mutable pointer arg, and arithmetic on a struct
        # field returns an immutable view — copy the base into a local first.
        for i in range(n):
            var dst = packed_buf + i * pv.row_bytes
            pack(indices + i * d, d, bits, dst)
    for i in range(n):
        pv.norms[i] = norms[i]
    return pv^


def from_pq_bytes(packed_flat: UnsafePointer[UInt8, MutExternalOrigin],
                  flat_nbytes: Int,
                  norms: UnsafePointer[Float32, MutExternalOrigin],
                  n: Int, d: Int, bits: Int) raises -> PackedVectors:
    """Build a PackedVectors from a `.pq`-style flat packed buffer.

    `.pq` packs `n*d` values as a single stream. For row-aligned widths
    this is the same layout as PackedVectors._packed and we copy verbatim;
    for row-padded widths we unpack the stream and repack row-by-row.
    """
    var pv = PackedVectors(n, d, bits)
    var packed_buf = pv.packed
    if pv.row_aligned:
        if flat_nbytes != n * pv.row_bytes:
            raise Error("from_pq_bytes: flat_nbytes != n * row_bytes (row-aligned)")
        for i in range(n * pv.row_bytes):
            packed_buf[i] = packed_flat[i]
    else:
        var unpacked = alloc[UInt8](n * d)
        unpack(packed_flat, n * d, bits, unpacked)
        for i in range(n):
            var dst = packed_buf + i * pv.row_bytes
            pack(unpacked + i * d, d, bits, dst)
        unpacked.free()
    for i in range(n):
        pv.norms[i] = norms[i]
    return pv^


def unpack_rows(pv: PackedVectors, start: Int, end: Int,
                mut out: UnsafePointer[UInt8, MutExternalOrigin]) raises:
    """Unpack rows [start, end) into `out` (size `(end - start) * d` uint8).

    Mirrors `PackedVectors.unpack_rows` in `remex/core.py`.
    """
    if start < 0 or end > pv.n or start > end:
        raise Error("unpack_rows: invalid [start, end)")
    var n_rows = end - start
    if pv.row_aligned:
        # Contiguous flat slice — single unpack call.
        var src = pv.packed + start * pv.row_bytes
        unpack(src, n_rows * pv.d, pv.bits, out)
    else:
        for i in range(n_rows):
            var src = pv.packed + (start + i) * pv.row_bytes
            var dst = out + i * pv.d
            unpack(src, pv.d, pv.bits, dst)


def unpack_all(pv: PackedVectors,
               mut out: UnsafePointer[UInt8, MutExternalOrigin]) raises:
    """Unpack all rows into `out` (size `n * d` uint8)."""
    unpack_rows(pv, 0, pv.n, out)


def unpack_at(pv: PackedVectors, idx: Int,
              mut out: UnsafePointer[UInt8, MutExternalOrigin]) raises:
    """Unpack a single row `idx` into `out` (size `d` uint8)."""
    if idx < 0 or idx >= pv.n:
        raise Error("unpack_at: index out of range")
    var src = pv.packed + idx * pv.row_bytes
    unpack(src, pv.d, pv.bits, out)


def at_precision(pv: PackedVectors, target_bits: Int) raises -> PackedVectors:
    """Derive a lower-bit PackedVectors via Matryoshka right-shift.

    Mirrors `PackedVectors.at_precision`: unpacks rows in chunks, shifts
    by `bits - target_bits`, and repacks at `target_bits`. When
    `target_bits == pv.bits` returns a fresh copy (Mojo lacks shared
    ownership, so we don't alias the input buffer).
    """
    if target_bits < 1 or target_bits > pv.bits:
        raise Error("at_precision: target_bits must be 1..bits")

    if target_bits == pv.bits:
        # Copy: caller owns the result independently.
        var same = PackedVectors(pv.n, pv.d, pv.bits)
        for i in range(pv.n * pv.row_bytes):
            same.packed[i] = pv.packed[i]
        for i in range(pv.n):
            same.norms[i] = pv.norms[i]
        return same^

    var shift = pv.bits - target_bits
    var pv_out = PackedVectors(pv.n, pv.d, target_bits)
    var out_buf = pv_out.packed
    var chunk = 4096
    var unpacked = alloc[UInt8](chunk * pv.d)
    var shifted = alloc[UInt8](chunk * pv.d)

    var start = 0
    while start < pv.n:
        var end = start + chunk
        if end > pv.n:
            end = pv.n
        var n_rows = end - start

        unpack_rows(pv, start, end, unpacked)
        for i in range(n_rows * pv.d):
            shifted[i] = unpacked[i] >> UInt8(shift)

        if pv_out.row_aligned:
            var dst = out_buf + start * pv_out.row_bytes
            pack(shifted, n_rows * pv.d, target_bits, dst)
        else:
            for i in range(n_rows):
                var dst = out_buf + (start + i) * pv_out.row_bytes
                pack(shifted + i * pv.d, pv.d, target_bits, dst)
        start = end

    for i in range(pv.n):
        pv_out.norms[i] = pv.norms[i]

    unpacked.free()
    shifted.free()
    return pv_out^
