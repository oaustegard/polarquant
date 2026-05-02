"""Test Lloyd-Max codebook against expected properties + Python parity."""

from std.testing import assert_true
from std.math import sqrt
from src.codebook import lloyd_max_codebook


def _abs(x: Float32) -> Float32:
    return -x if x < Float32(0.0) else x


def test_symmetric_4bit() raises:
    """Centroids of an even Lloyd-Max codebook are symmetric around 0."""
    var cb = lloyd_max_codebook(384, 4)
    for i in range(8):
        var left = cb.get_centroid(i)
        var right = cb.get_centroid(15 - i)
        var sym_err = _abs(left + right)
        assert_true(sym_err < Float32(1e-5))
    print("test_symmetric_4bit: ok")


def test_monotone_4bit() raises:
    """Centroids must be strictly increasing."""
    var cb = lloyd_max_codebook(384, 4)
    for i in range(15):
        assert_true(cb.get_centroid(i) < cb.get_centroid(i + 1))
    print("test_monotone_4bit: ok")


def test_boundaries_are_midpoints() raises:
    var cb = lloyd_max_codebook(384, 3)
    for i in range(7):
        var mid = Float32(0.5) * (cb.get_centroid(i) + cb.get_centroid(i + 1))
        assert_true(_abs(cb.get_boundary(i) - mid) < Float32(1e-7))
    print("test_boundaries_are_midpoints: ok")


def test_shape() raises:
    """1, 2, 4, 8 bits → 2, 4, 16, 256 levels."""
    var cb1 = lloyd_max_codebook(384, 1)
    assert_true(cb1.n_levels == 2)
    var cb4 = lloyd_max_codebook(384, 4)
    assert_true(cb4.n_levels == 16)
    var cb8 = lloyd_max_codebook(384, 8)
    assert_true(cb8.n_levels == 256)
    print("test_shape: ok")


def test_known_values() raises:
    """Spot-check 1-bit codebook: ±E[|N(0,sigma)|] = ±sigma*sqrt(2/pi)."""
    var d = 384
    var sigma = Float32(1.0) / Float32(sqrt(Float64(d)))
    var expected_mag = sigma * Float32(sqrt(Float64(2.0) / Float64(3.141592653589793)))
    var cb = lloyd_max_codebook(d, 1)
    var c0: Float32 = cb.get_centroid(0)
    var c1: Float32 = cb.get_centroid(1)
    var lhs1 = _abs(c0 + expected_mag)
    var lhs2 = _abs(c1 - expected_mag)
    assert_true(lhs1 < Float32(1e-5))
    assert_true(lhs2 < Float32(1e-5))
    print("test_known_values: ok")


def main() raises:
    test_shape()
    test_symmetric_4bit()
    test_monotone_4bit()
    test_boundaries_are_midpoints()
    test_known_values()
    print("[test_codebook] all passed")
