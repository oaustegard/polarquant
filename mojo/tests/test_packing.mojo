"""Roundtrip tests for pack/unpack at all supported widths."""

from std.testing import assert_equal, assert_true
from std.memory import alloc
from src.packing import pack, unpack, packed_nbytes
from src.rng import Xoshiro256pp


def _roundtrip(n: Int, bits: Int, seed: UInt64) raises:
    var rng = Xoshiro256pp(seed)
    var mask = UInt64((1 << bits) - 1) if bits < 8 else UInt64(0xFF)

    var inp = alloc[UInt8](n)
    for i in range(n):
        inp[i] = UInt8(rng.next_u64() & mask)

    var nb = packed_nbytes(n, bits)
    var packed = alloc[UInt8](nb)
    pack(inp, n, bits, packed)

    var out = alloc[UInt8](n)
    unpack(packed, n, bits, out)

    for i in range(n):
        assert_equal(Int(out[i]), Int(inp[i]))

    inp.free()
    packed.free()
    out.free()


def test_roundtrips() raises:
    var widths = [1, 2, 3, 4, 8]
    var sizes = [1, 7, 8, 9, 16, 17, 100, 1000]
    for bw in widths:
        for n in sizes:
            _roundtrip(n, bw, 42)
        print("  bits =", bw, " roundtrip ok")


def test_packed_nbytes() raises:
    assert_equal(packed_nbytes(8, 8), 8)
    assert_equal(packed_nbytes(8, 4), 4)
    assert_equal(packed_nbytes(8, 2), 2)
    assert_equal(packed_nbytes(8, 1), 1)
    assert_equal(packed_nbytes(8, 3), 3)
    assert_equal(packed_nbytes(7, 4), 4)   # ceil(7/2)=4
    assert_equal(packed_nbytes(7, 1), 1)   # ceil(7/8)=1
    assert_equal(packed_nbytes(9, 3), 6)   # ceil(9/8)*3 = 2*3 = 6
    print("test_packed_nbytes: ok")


def main() raises:
    test_packed_nbytes()
    test_roundtrips()
    print("[test_packing] all passed")
