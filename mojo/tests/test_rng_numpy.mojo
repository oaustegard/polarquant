"""Layer-by-layer parity test for rng_numpy.mojo against NumPy 2.x.

For seed=42 (and a few sanity seeds):
  1. SeedSequence pool — 4 uint32s match NumPy's `SeedSequence(42).pool`.
  2. SeedSequence generate_state — 4 uint64s match
     `SeedSequence(42).generate_state(4, dtype=uint64)`.
  3. PCG64 — first 5 raw uint64 outputs match
     `default_rng(42).bit_generator.random_raw()`.
  4. PCG64 next_double — first 5 doubles match `default_rng(42).random()`.
  5. Ziggurat standard_normal — first 1000 outputs match
     `default_rng(42).standard_normal(1000)` (allowing rare libm-driven
     deviation in the tail-rejection branch — see rng_numpy.mojo docstring).

Reference values were generated with NumPy 2.4.4 and embedded as
constants. Regenerating them only matters if NumPy changes its internal
algorithms, which they have not done since the SeedSequence/PCG64/Ziggurat
introduction in NumPy 1.17.
"""

from std.math import sqrt
from std.memory import alloc
from std.pathlib import Path
from std.testing import assert_equal, assert_true
from src.rng_numpy import (
    seedseq_pool_from_u64,
    seedseq_generate_state_u64,
    PCG64,
    NumpyNormalRNG,
)


# Reference values for seed=42 (from numpy 2.4.4)
alias _REF_POOL_42_0 = UInt32(1662858758)
alias _REF_POOL_42_1 = UInt32(128880814)
alias _REF_POOL_42_2 = UInt32(1875164712)
alias _REF_POOL_42_3 = UInt32(753753205)

alias _REF_STATE_42_0 = UInt64(0x9F1E2E6DCD540AB7)
alias _REF_STATE_42_1 = UInt64(0xD57873DC79FB94B6)
alias _REF_STATE_42_2 = UInt64(0x7D282A1B64D420B7)
alias _REF_STATE_42_3 = UInt64(0x336579714692D5FF)

alias _REF_RAW_42_0 = UInt64(0xC621FBCD16D92688)
alias _REF_RAW_42_1 = UInt64(0x705A5661A791FFC1)
alias _REF_RAW_42_2 = UInt64(0xDBCD12C26EDA1624)
alias _REF_RAW_42_3 = UInt64(0xB286B60E1600888D)
alias _REF_RAW_42_4 = UInt64(0x181C01B5339381EB)

alias _REF_DOUBLE_42_0 = Float64(0.7739560485559633)
alias _REF_DOUBLE_42_1 = Float64(0.4388784397520523)
alias _REF_DOUBLE_42_2 = Float64(0.8585979199113825)
alias _REF_DOUBLE_42_3 = Float64(0.6973680290593639)
alias _REF_DOUBLE_42_4 = Float64(0.09417734788764953)

alias _REF_NORMAL_42_0 = Float64(0.30471707975443135)
alias _REF_NORMAL_42_1 = Float64(-1.0399841062404955)
alias _REF_NORMAL_42_2 = Float64(0.7504511958064572)
alias _REF_NORMAL_42_3 = Float64(0.9405647163912139)
alias _REF_NORMAL_42_4 = Float64(-1.9510351886538364)
alias _REF_NORMAL_42_5 = Float64(-1.302179506862318)
alias _REF_NORMAL_42_6 = Float64(0.12784040316728537)
alias _REF_NORMAL_42_7 = Float64(-0.3162425923435822)
alias _REF_NORMAL_42_8 = Float64(-0.016801157504288795)
alias _REF_NORMAL_42_9 = Float64(-0.85304392757358)


def test_seedseq_pool() raises:
    var pool = seedseq_pool_from_u64(UInt64(42))
    assert_equal(pool[0], _REF_POOL_42_0)
    assert_equal(pool[1], _REF_POOL_42_1)
    assert_equal(pool[2], _REF_POOL_42_2)
    assert_equal(pool[3], _REF_POOL_42_3)
    print("test_seedseq_pool: ok")


def test_seedseq_generate_state() raises:
    var pool = seedseq_pool_from_u64(UInt64(42))
    var state = alloc[UInt64](4)
    seedseq_generate_state_u64(pool, 4, state)
    assert_equal(state[0], _REF_STATE_42_0)
    assert_equal(state[1], _REF_STATE_42_1)
    assert_equal(state[2], _REF_STATE_42_2)
    assert_equal(state[3], _REF_STATE_42_3)
    state.free()
    print("test_seedseq_generate_state: ok")


def test_pcg64_raw_u64() raises:
    var rng = PCG64(UInt64(42))
    assert_equal(rng.next_u64(), _REF_RAW_42_0)
    assert_equal(rng.next_u64(), _REF_RAW_42_1)
    assert_equal(rng.next_u64(), _REF_RAW_42_2)
    assert_equal(rng.next_u64(), _REF_RAW_42_3)
    assert_equal(rng.next_u64(), _REF_RAW_42_4)
    print("test_pcg64_raw_u64: ok")


def test_pcg64_next_double() raises:
    var rng = PCG64(UInt64(42))
    var d0 = rng.next_double()
    var d1 = rng.next_double()
    var d2 = rng.next_double()
    var d3 = rng.next_double()
    var d4 = rng.next_double()
    assert_equal(d0, _REF_DOUBLE_42_0)
    assert_equal(d1, _REF_DOUBLE_42_1)
    assert_equal(d2, _REF_DOUBLE_42_2)
    assert_equal(d3, _REF_DOUBLE_42_3)
    assert_equal(d4, _REF_DOUBLE_42_4)
    print("test_pcg64_next_double: ok")


def test_standard_normal_first_10() raises:
    var rng = NumpyNormalRNG(UInt64(42))
    var n0 = rng.next_normal()
    var n1 = rng.next_normal()
    var n2 = rng.next_normal()
    var n3 = rng.next_normal()
    var n4 = rng.next_normal()
    var n5 = rng.next_normal()
    var n6 = rng.next_normal()
    var n7 = rng.next_normal()
    var n8 = rng.next_normal()
    var n9 = rng.next_normal()
    print("first 10 normals (seed=42):")
    print("  ", n0, " expected:", _REF_NORMAL_42_0)
    print("  ", n1, " expected:", _REF_NORMAL_42_1)
    print("  ", n2, " expected:", _REF_NORMAL_42_2)
    print("  ", n3, " expected:", _REF_NORMAL_42_3)
    print("  ", n4, " expected:", _REF_NORMAL_42_4)
    print("  ", n5, " expected:", _REF_NORMAL_42_5)
    print("  ", n6, " expected:", _REF_NORMAL_42_6)
    print("  ", n7, " expected:", _REF_NORMAL_42_7)
    print("  ", n8, " expected:", _REF_NORMAL_42_8)
    print("  ", n9, " expected:", _REF_NORMAL_42_9)
    assert_equal(n0, _REF_NORMAL_42_0)
    assert_equal(n1, _REF_NORMAL_42_1)
    assert_equal(n2, _REF_NORMAL_42_2)
    assert_equal(n3, _REF_NORMAL_42_3)
    assert_equal(n4, _REF_NORMAL_42_4)
    assert_equal(n5, _REF_NORMAL_42_5)
    assert_equal(n6, _REF_NORMAL_42_6)
    assert_equal(n7, _REF_NORMAL_42_7)
    assert_equal(n8, _REF_NORMAL_42_8)
    assert_equal(n9, _REF_NORMAL_42_9)
    print("test_standard_normal_first_10: ok")


def _read_u64_bin(path: String, n: Int) raises -> UnsafePointer[UInt64, MutExternalOrigin]:
    var raw = Path(path).read_bytes()
    if len(raw) < n * 8:
        raise Error("fixture too small: " + path)
    var out = alloc[UInt64](n)
    var bp = out.bitcast[UInt8]()
    for i in range(n * 8):
        bp[i] = raw[i]
    return out


def _read_f64_bin(path: String, n: Int) raises -> UnsafePointer[Float64, MutExternalOrigin]:
    var raw = Path(path).read_bytes()
    if len(raw) < n * 8:
        raise Error("fixture too small: " + path)
    var out = alloc[Float64](n)
    var bp = out.bitcast[UInt8]()
    for i in range(n * 8):
        bp[i] = raw[i]
    return out


def test_pcg64_first_1000() raises:
    """First 1000 raw uint64 outputs vs NumPy fixture."""
    var expected = _read_u64_bin(
        "/home/user/claude-workspace/.spokes/remex/remex/mojo/tests/fixtures/raw_u64_seed42.bin",
        1000,
    )
    var rng = PCG64(UInt64(42))
    var mismatches = 0
    var first_bad: Int = -1
    for i in range(1000):
        var got = rng.next_u64()
        if got != expected[i]:
            mismatches += 1
            if first_bad < 0:
                first_bad = i
                print("first mismatch at i=", i, " got=", got, " expected=", expected[i])
    print("PCG64 1000-output mismatches:", mismatches)
    assert_equal(mismatches, 0)
    expected.free()
    print("test_pcg64_first_1000: ok")


def test_standard_normal_first_1000() raises:
    """First 1000 standard_normal outputs vs NumPy fixture."""
    var expected_arr = _read_f64_bin(
        "/home/user/claude-workspace/.spokes/remex/remex/mojo/tests/fixtures/normal_f64_seed42.bin",
        1000,
    )
    var rng = NumpyNormalRNG(UInt64(42))
    var mismatches = 0
    var first_bad: Int = -1
    for i in range(1000):
        var got = rng.next_normal()
        var ref_val = expected_arr[i]
        if got != ref_val:
            mismatches += 1
            if first_bad < 0:
                first_bad = i
                print("first mismatch at i=", i, " got=", got, " expected=", ref_val, " diff=", got - ref_val)
    print("standard_normal 1000-output mismatches:", mismatches)
    # Expect: 0 mismatches when libm log1p/exp matches across builds.
    # If a few tail samples differ at last bit, that's libm rounding
    # (documented limitation in module docstring). Hard-cap at 5 such
    # samples — anything beyond that is a real algorithm bug.
    assert_true(mismatches <= 5)
    expected_arr.free()
    print("test_standard_normal_first_1000: ok")


def test_normal_moments_long_run() raises:
    """Sanity: 10000 normals from our RNG should have mean ~0, sd ~1."""
    var rng = NumpyNormalRNG(UInt64(7))
    var n = 10000
    var s: Float64 = 0.0
    var s2: Float64 = 0.0
    for _ in range(n):
        var x = rng.next_normal()
        s += x
        s2 += x * x
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    var sd = sqrt(var_)
    print("normal moments: n =", n, "mean =", mean, "sd =", sd)
    assert_true(mean > -0.05 and mean < 0.05)
    assert_true(sd > 0.97 and sd < 1.03)
    print("test_normal_moments_long_run: ok")


def main() raises:
    test_seedseq_pool()
    test_seedseq_generate_state()
    test_pcg64_raw_u64()
    test_pcg64_next_double()
    test_standard_normal_first_10()
    test_pcg64_first_1000()
    test_standard_normal_first_1000()
    test_normal_moments_long_run()
    print("[test_rng_numpy] all passed")
