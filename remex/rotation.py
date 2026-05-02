"""Random orthogonal rotation via Haar-distributed matrices.

Uses an explicit Householder QR with a fixed reflector convention so the
output is bit-reproducible across BLAS builds and matches the Mojo port's
encode path byte-for-byte (issue #40). The previous implementation
delegated to ``np.linalg.qr``, which calls LAPACK ``dgeqrf``; LAPACK QR
is not bit-deterministic across MKL/OpenBLAS builds or threading modes,
which made `--seed`-based reproducibility impossible end-to-end.
"""

import numpy as np


def _householder_qr(A: np.ndarray) -> np.ndarray:
    """In-place Householder QR; returns Q. After the call A holds R.

    Reflector convention (must match ``mojo/src/rotation.mojo``):
      - ``alpha = -sign(A[k,k]) * ||A[k:, k]||`` with ``sign(0) = +1``
      - ``v = A[k:, k] - alpha * e_1``, normalized
      - Apply ``H = I - 2 v v^T`` to ``A[k:, k:]`` from the left and to
        ``Q[:, k:]`` from the right
    """
    n = A.shape[0]
    Q = np.eye(n, dtype=np.float64)
    for k in range(n - 1):
        col = A[k:, k]
        col_norm = float(np.sqrt(np.dot(col, col)))
        if col_norm == 0.0:
            continue
        sign_akk = -1.0 if A[k, k] < 0.0 else 1.0
        alpha = -sign_akk * col_norm

        v = col.copy()
        v[0] -= alpha
        v_norm = float(np.sqrt(np.dot(v, v)))
        if v_norm == 0.0:
            continue
        v /= v_norm

        # H = I - 2 v v^T applied to A[k:, k:] from the left
        # A[k:, k:] -= 2 * outer(v, v.T @ A[k:, k:])
        sub = A[k:, k:]
        sub -= 2.0 * np.outer(v, v @ sub)

        # Same H applied to Q[:, k:] from the right
        # Q[:, k:] -= 2 * outer(Q[:, k:] @ v, v)
        Qsub = Q[:, k:]
        Qsub -= 2.0 * np.outer(Qsub @ v, v)
    return Q


def haar_rotation(d: int, seed: int = 42) -> np.ndarray:
    """
    Generate a Haar-distributed random orthogonal matrix.

    Pipeline (matches the Mojo port for byte-identical encode parity):
      1. Sample G ~ N(0, 1) of shape (d, d) in float64 via NumPy's
         default RNG (PCG64 + SeedSequence + Ziggurat).
      2. Run explicit Householder QR with the fixed reflector convention
         documented in ``_householder_qr``.
      3. Apply Mezzadri sign correction so Q is uniformly distributed
         on O(d) (the Haar measure).
      4. Cast Q to float32.

    Args:
        d: Matrix dimension.
        seed: Random seed for reproducibility.

    Returns:
        Q: (d, d) float32 orthogonal matrix, Q @ Q.T ~ I.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))  # float64 by default
    Q = _householder_qr(A)

    # Mezzadri sign correction: ensure diag(R) > 0 so Q is Haar-distributed.
    diag = np.diagonal(A).copy()
    sign_flip = diag < 0.0
    if sign_flip.any():
        Q[:, sign_flip] = -Q[:, sign_flip]

    return Q.astype(np.float32)
