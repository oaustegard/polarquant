"""Test rotation: orthogonality + determinism."""

from std.math import sqrt
from std.testing import assert_true
from std.memory import alloc
from src.rotation import haar_rotation, matvec, matvec_T
from src.matrix import Matrix


def _abs(x: Float32) -> Float32:
    return -x if x < Float32(0.0) else x


def _orthogonality_error(M: Matrix) -> Float32:
    """max |M.T @ M - I|."""
    var d = M.rows
    var max_err: Float32 = Float32(0.0)
    for i in range(d):
        for j in range(d):
            var s: Float32 = Float32(0.0)
            for k in range(d):
                s += M.get(k, i) * M.get(k, j)
            var target: Float32 = Float32(1.0) if i == j else Float32(0.0)
            var err = _abs(s - target)
            if err > max_err:
                max_err = err
    return max_err


def test_orthogonal() raises:
    var R = haar_rotation(8, 42)
    var err = _orthogonality_error(R)
    print("orthogonality err (d=8) =", err)
    assert_true(err < 1e-5)
    print("test_orthogonal: ok")


def test_determinism() raises:
    var R1 = haar_rotation(16, 7)
    var R2 = haar_rotation(16, 7)
    var max_diff: Float32 = Float32(0.0)
    for i in range(16 * 16):
        var d = _abs(R1.data[i] - R2.data[i])
        if d > max_diff:
            max_diff = d
    print("determinism max_diff =", max_diff)
    assert_true(max_diff == Float32(0.0))
    print("test_determinism: ok")


def test_norm_preserving() raises:
    var d = 32
    var R = haar_rotation(d, 1)
    var x = alloc[Float32](d)
    var y = alloc[Float32](d)
    for i in range(d):
        x[i] = Float32(i + 1) * Float32(0.1) - Float32(1.0)
    matvec(R, x, y)
    var nx: Float32 = Float32(0.0)
    var ny: Float32 = Float32(0.0)
    for i in range(d):
        nx += x[i] * x[i]
        ny += y[i] * y[i]
    print("||x||^2 =", nx, "||Rx||^2 =", ny)
    assert_true(_abs(nx - ny) < Float32(1e-3))
    x.free()
    y.free()
    print("test_norm_preserving: ok")


def main() raises:
    test_orthogonal()
    test_determinism()
    test_norm_preserving()
    print("[test_rotation] all passed")
