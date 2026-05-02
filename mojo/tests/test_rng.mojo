"""Test rng.mojo: determinism + basic statistical sanity."""

from src.rng import Xoshiro256pp
from std.math import sqrt
from std.testing import assert_equal, assert_true


def test_determinism() raises:
    var r1 = Xoshiro256pp(42)
    var r2 = Xoshiro256pp(42)
    for _ in range(1000):
        assert_equal(r1.next_u64(), r2.next_u64())
    print("test_determinism: ok")


def test_uniform_range() raises:
    var r = Xoshiro256pp(7)
    for _ in range(10000):
        var u = r.next_uniform()
        assert_true(u >= 0.0 and u < 1.0)
    print("test_uniform_range: ok")


def test_normal_moments() raises:
    var r = Xoshiro256pp(123)
    var n = 200000
    var s: Float64 = 0.0
    var s2: Float64 = 0.0
    for _ in range(n):
        var x = r.next_normal()
        s += x
        s2 += x * x
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    var sd = sqrt(var_)
    print("normal: n =", n, "mean =", mean, "sd =", sd)
    # ~3 sigma bounds for n=200k:
    assert_true(mean > -0.02 and mean < 0.02)
    assert_true(sd > 0.99 and sd < 1.01)
    print("test_normal_moments: ok")


def main() raises:
    test_determinism()
    test_uniform_range()
    test_normal_moments()
    print("[test_rng] all passed")
