"""Haar-distributed random orthogonal matrix via Householder QR.

Mirrors `remex.rotation.haar_rotation`:
  1. Sample A ~ N(0, 1) of shape (d, d)
  2. QR decompose
  3. Sign-correct: Q[:, j] *= sign(R[j, j])

The resulting Q is uniformly distributed on O(d) (Mezzadri 2007).
"""

from std.math import sqrt
from std.memory import alloc, UnsafePointer
from src.rng import Xoshiro256pp
from src.rng_numpy import NumpyNormalRNG
from src.matrix import Matrix, MatrixF64


def _norm2(p: UnsafePointer[Float64, MutExternalOrigin], n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        s += p[i] * p[i]
    return sqrt(s)


def _householder_qr(mut A: MatrixF64, mut Q: MatrixF64):
    """In-place QR. After call: A holds R (upper triangular), Q holds Q."""
    var n = A.rows
    # Initialize Q = I
    for i in range(n):
        for j in range(n):
            Q.set(i, j, Float64(1.0) if i == j else Float64(0.0))

    for k in range(n - 1):
        # Compute alpha = -sign(A[k,k]) * ||A[k:, k]||
        var col_norm_sq: Float64 = 0.0
        for i in range(k, n):
            var v = A.get(i, k)
            col_norm_sq += v * v
        var col_norm = sqrt(col_norm_sq)
        if col_norm == 0.0:
            continue
        var sign_akk: Float64 = -1.0 if A.get(k, k) < 0.0 else 1.0
        var alpha = -sign_akk * col_norm

        # v = A[k:, k] - alpha * e1
        # store v in a temp column-vector
        var v_len = n - k
        var vp = alloc[Float64](v_len)
        for i in range(v_len):
            vp[i] = A.get(k + i, k)
        vp[0] -= alpha

        var v_norm = _norm2(vp, v_len)
        if v_norm == 0.0:
            vp.free()
            continue
        for i in range(v_len):
            vp[i] /= v_norm

        # Apply H = I - 2 v v^T to A[k:, k:] from the left:
        # for each column j >= k: A[k:, j] -= 2 v (v^T A[k:, j])
        for j in range(k, n):
            var dot: Float64 = 0.0
            for i in range(v_len):
                dot += vp[i] * A.get(k + i, j)
            var two_dot = 2.0 * dot
            for i in range(v_len):
                A.set(k + i, j, A.get(k + i, j) - two_dot * vp[i])

        # Apply H to Q from the right: Q[:, k:] -= 2 (Q[:, k:] v) v^T
        for i in range(n):
            var dot: Float64 = 0.0
            for j in range(v_len):
                dot += Q.get(i, k + j) * vp[j]
            var two_dot = 2.0 * dot
            for j in range(v_len):
                Q.set(i, k + j, Q.get(i, k + j) - two_dot * vp[j])

        vp.free()


def haar_rotation(d: Int, seed: UInt64) -> Matrix:
    """Haar-distributed (d, d) orthogonal matrix using xoshiro256++ + Marsaglia.

    NOT bit-identical to Python `Quantizer(seed=S).R` — produces a valid
    Haar sample but from a different Gaussian stream. Use
    `haar_rotation_numpy` for byte parity with Python.
    """
    var rng = Xoshiro256pp(seed)
    var A = MatrixF64(d, d)
    for i in range(d):
        for j in range(d):
            A.set(i, j, rng.next_normal())

    var Q = MatrixF64(d, d)
    _householder_qr(A, Q)

    # Sign-correct: Q[:, j] *= sign(R[j, j])
    for j in range(d):
        var diag = A.get(j, j)
        if diag < 0.0:
            for i in range(d):
                Q.set(i, j, -Q.get(i, j))

    return Q.to_float32()


def haar_rotation_numpy(d: Int, seed: UInt64) -> Matrix:
    """Haar-distributed (d, d) orthogonal matrix matching Python `Quantizer(seed=S)`.

    Generates G via the NumPy-compatible RNG (PCG64 + SeedSequence + Ziggurat),
    then runs the same Householder QR + Mezzadri sign correction as
    `haar_rotation`. End-to-end this matches Python's `remex.haar_rotation(d, seed)`
    bit-for-bit at float32 (modulo rare libm tail-rejection rounding).
    """
    var rng = NumpyNormalRNG(seed)
    var A = MatrixF64(d, d)
    # NumPy fills row-major: A[i, j] is the (i*d + j)-th draw.
    for i in range(d):
        for j in range(d):
            A.set(i, j, rng.next_normal())

    var Q = MatrixF64(d, d)
    _householder_qr(A, Q)

    for j in range(d):
        var diag = A.get(j, j)
        if diag < 0.0:
            for i in range(d):
                Q.set(i, j, -Q.get(i, j))

    return Q.to_float32()


def matvec(M: Matrix, x: UnsafePointer[Float32, MutExternalOrigin],
           mut out_buf: UnsafePointer[Float32, MutExternalOrigin]):
    """y = M @ x, where M is (rows, cols) and x is length cols."""
    for i in range(M.rows):
        var s: Float32 = Float32(0.0)
        var base = i * M.cols
        for j in range(M.cols):
            s += M.data[base + j] * x[j]
        out_buf[i] = s


def matvec_T(M: Matrix, x: UnsafePointer[Float32, MutExternalOrigin],
             mut out_buf: UnsafePointer[Float32, MutExternalOrigin]):
    """y = M.T @ x. (i.e. dot product of x with each column of M.)"""
    for j in range(M.cols):
        out_buf[j] = Float32(0.0)
    for i in range(M.rows):
        var base = i * M.cols
        var xi = x[i]
        for j in range(M.cols):
            out_buf[j] += M.data[base + j] * xi
