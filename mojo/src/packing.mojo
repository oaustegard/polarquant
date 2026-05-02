"""Bit-packing for sub-byte indices. Mirrors `remex/packing.py` exactly.

Supported widths: 1, 2, 3, 4, 8 bits. 5-7 are rejected.

For 1-bit, MSB-first ordering matches `np.packbits` default.
"""

from std.memory import alloc, UnsafePointer


def packed_nbytes(n_values: Int, bits: Int) raises -> Int:
    """Bytes required to pack `n_values` indices at the given width."""
    if bits == 8:
        return n_values
    if bits == 4:
        return (n_values + 1) // 2
    if bits == 2:
        return (n_values + 3) // 4
    if bits == 1:
        return (n_values + 7) // 8
    if bits == 3:
        return ((n_values + 7) // 8) * 3
    raise Error("bits must be 1, 2, 3, 4, or 8 (5-7 unsupported)")


def pack(values: UnsafePointer[UInt8, MutExternalOrigin], n: Int, bits: Int,
         mut out: UnsafePointer[UInt8, MutExternalOrigin]) raises:
    """Pack `n` uint8 indices from `values` into `out` (size = packed_nbytes(n, bits))."""
    if bits == 8:
        for i in range(n):
            out[i] = values[i]
        return

    if bits == 4:
        var nb = (n + 1) // 2
        for i in range(nb):
            var hi = values[2 * i]
            var lo: UInt8 = UInt8(0)
            if 2 * i + 1 < n:
                lo = values[2 * i + 1]
            out[i] = (hi << 4) | lo
        return

    if bits == 2:
        var nb = (n + 3) // 4
        for i in range(nb):
            var v0 = values[4 * i] if 4 * i < n else UInt8(0)
            var v1 = values[4 * i + 1] if 4 * i + 1 < n else UInt8(0)
            var v2 = values[4 * i + 2] if 4 * i + 2 < n else UInt8(0)
            var v3 = values[4 * i + 3] if 4 * i + 3 < n else UInt8(0)
            out[i] = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3
        return

    if bits == 1:
        var nb = (n + 7) // 8
        for i in range(nb):
            var b: UInt8 = UInt8(0)
            for k in range(8):
                var idx = 8 * i + k
                if idx < n:
                    b |= (values[idx] & UInt8(1)) << UInt8(7 - k)
            out[i] = b
        return

    if bits == 3:
        var ng = (n + 7) // 8     # number of 8-value groups
        for g in range(ng):
            var v = InlineArray[UInt8, 8](fill=UInt8(0))
            for k in range(8):
                var idx = 8 * g + k
                if idx < n:
                    v[k] = values[idx]
            # Byte 0: [v0:3][v1:3][v2:2hi]
            out[3 * g] = (v[0] << 5) | (v[1] << 2) | (v[2] >> 1)
            # Byte 1: [v2:1lo][v3:3][v4:3][v5:1hi]
            out[3 * g + 1] = ((v[2] & UInt8(1)) << 7) | (v[3] << 4) | (v[4] << 1) | (v[5] >> 2)
            # Byte 2: [v5:2lo][v6:3][v7:3]
            out[3 * g + 2] = ((v[5] & UInt8(3)) << 6) | (v[6] << 3) | v[7]
        return

    raise Error("bits must be 1, 2, 3, 4, or 8 (5-7 unsupported)")


def unpack(packed: UnsafePointer[UInt8, MutExternalOrigin],
           n_values: Int, bits: Int,
           mut out: UnsafePointer[UInt8, MutExternalOrigin]) raises:
    """Unpack `n_values` indices from `packed` into `out`."""
    if bits == 8:
        for i in range(n_values):
            out[i] = packed[i]
        return

    if bits == 4:
        for i in range(n_values):
            var byte = packed[i // 2]
            if i % 2 == 0:
                out[i] = (byte >> 4) & UInt8(0x0F)
            else:
                out[i] = byte & UInt8(0x0F)
        return

    if bits == 2:
        for i in range(n_values):
            var byte = packed[i // 4]
            var slot = i % 4
            if slot == 0:
                out[i] = (byte >> 6) & UInt8(0x03)
            elif slot == 1:
                out[i] = (byte >> 4) & UInt8(0x03)
            elif slot == 2:
                out[i] = (byte >> 2) & UInt8(0x03)
            else:
                out[i] = byte & UInt8(0x03)
        return

    if bits == 1:
        for i in range(n_values):
            var byte = packed[i // 8]
            var bit = 7 - (i % 8)
            out[i] = (byte >> UInt8(bit)) & UInt8(1)
        return

    if bits == 3:
        var ng = (n_values + 7) // 8
        for g in range(ng):
            var b0 = packed[3 * g]
            var b1 = packed[3 * g + 1]
            var b2 = packed[3 * g + 2]
            var v = InlineArray[UInt8, 8](fill=UInt8(0))
            v[0] = (b0 >> 5) & UInt8(0x07)
            v[1] = (b0 >> 2) & UInt8(0x07)
            v[2] = ((b0 & UInt8(0x03)) << 1) | ((b1 >> 7) & UInt8(0x01))
            v[3] = (b1 >> 4) & UInt8(0x07)
            v[4] = (b1 >> 1) & UInt8(0x07)
            v[5] = ((b1 & UInt8(0x01)) << 2) | ((b2 >> 6) & UInt8(0x03))
            v[6] = (b2 >> 3) & UInt8(0x07)
            v[7] = b2 & UInt8(0x07)
            for k in range(8):
                var idx = 8 * g + k
                if idx < n_values:
                    out[idx] = v[k]
        return

    raise Error("bits must be 1, 2, 3, 4, or 8 (5-7 unsupported)")
