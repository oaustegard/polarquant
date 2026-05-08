"""GPU encode kernel for `polarquant encode --device gpu`.

First-cut single-source kernel that runs on Apple Metal, NVIDIA, and AMD
via Mojo 1.0's GPU dispatch. Mirrors `src/quantizer.mojo::encode_batch`
contract and aims for byte-identical output vs Python.

## Strategy for byte parity

`encode_batch` does three things per row:
  1. norm = sqrt(sum_j X[i,j]^2)       in float64, then cast to float32
  2. rotated[k] = (sum_j R[k,j] * X[i,j]) * (1/norm)
  3. indices[k] = searchsorted_left(boundaries, rotated[k])

Step 1 is done on CPU. The float64 reduction order matches Python's
`np.sum(X.astype(f64)**2)` exactly, so norms are byte-identical. Doing
the same on GPU would require either fp64 (slow on Apple silicon) or a
parallel reduction (changes order). CPU norms are O(n*d) — negligible
next to the O(n*d^2) rotation cost.

Steps 2+3 are fused in one kernel: one thread per (row, k) output. Each
thread accumulates its dot product left-to-right, matches CPU FP order
(`_dot_f32` style — no SIMD vectorization in the GPU path). Borderline
coordinates near a boundary may still flip due to operand-order
differences in the FMA pipeline; the test (test_gpu_encode.mojo) starts
with a strict byte-equal assert and is documented to fall back to a
fraction-within-tolerance check if real silicon disagrees.

## Memory access optimization

The kernel receives R_T, the column-major transpose of R (stored row-major
as R_T[j, k] = R[k, j]).  This turns what was a strided-by-d access into
a coalesced access: adjacent threads in the x-dimension (adjacent `col`
values) read R_T[j * d + col], which are contiguous in memory.

Without transposition:
  thread `col` reads R[col * d + j] — between adjacent threads the
  addresses differ by d = 384 floats = 1536 bytes → 32 cache misses per
  warp step.

With R_T:
  thread `col` reads R_T[j * d + col] — adjacent threads differ by one
  float = 4 bytes → one cache line covers an entire 16-thread group.

R_T is computed on the host once per `gpu_encode_batch` call via an O(d²)
transpose (< 600 µs at d=384) and sent to the device in place of R.
X reads are already broadcast-efficient (all threads in a warp share the
same row and therefore the same X[row, j] at each j step).
"""

from std.math import sqrt
from std.memory import alloc, UnsafePointer
from std.gpu import block_idx, block_dim, thread_idx
from std.gpu.host import DeviceContext

from src.quantizer import Quantizer


def _encode_kernel(
    X: UnsafePointer[Float32, MutAnyOrigin],
    R_T: UnsafePointer[Float32, MutAnyOrigin],  # R transposed: R_T[j*d+k] = R[k*d+j]
    inv_norms: UnsafePointer[Float32, MutAnyOrigin],
    boundaries: UnsafePointer[Float32, MutAnyOrigin],
    indices_out: UnsafePointer[UInt8, MutAnyOrigin],
    n: Int32,
    d: Int32,
    n_b: Int32,
):
    var col = Int32(block_idx.x * block_dim.x + thread_idx.x)
    var row = Int32(block_idx.y * block_dim.y + thread_idx.y)
    if row >= n or col >= d:
        return

    var ri = Int(row)
    var ci = Int(col)
    var di = Int(d)

    # Rotation matvec — left-to-right accumulation matches CPU `_dot_f32`.
    # R_T[j * d + ci] = R[ci * d + j]: reading R_T with fixed j and varying ci
    # (across warp threads) gives coalesced access — adjacent ci addresses
    # differ by 4 bytes, vs 384*4 bytes for the original R[ci*d+j] layout.
    var rot: Float32 = 0.0
    for j in range(di):
        rot += R_T[j * di + ci] * X[ri * di + j]
    rot *= inv_norms[ri]

    # searchsorted_left: smallest i such that rot < boundaries[i]; else n_b.
    var lo: Int32 = 0
    var hi: Int32 = n_b
    while lo < hi:
        var mid = (lo + hi) >> Int32(1)
        if rot < boundaries[Int(mid)]:
            hi = mid
        else:
            lo = mid + Int32(1)
    indices_out[ri * di + ci] = UInt8(Int(lo))


def _sumsq_f64(x: UnsafePointer[Float32, MutExternalOrigin], d: Int) -> Float64:
    """Float64 sum-of-squares matching `np.sum(X.astype(f64)**2)` order."""
    var s: Float64 = 0.0
    for j in range(d):
        var xj = Float64(x[j])
        s += xj * xj
    return s


def gpu_encode_batch(q: Quantizer,
                     X: UnsafePointer[Float32, MutExternalOrigin],
                     n: Int,
                     mut indices_out: UnsafePointer[UInt8, MutExternalOrigin],
                     mut norms_out: UnsafePointer[Float32, MutExternalOrigin]) raises:
    """GPU mirror of `encode_batch`. Same contract as the CPU path."""
    var d = q.d
    var n_levels = q.cb.n_levels
    var n_b = n_levels - 1

    # 1. Norms on CPU — float64 sum-of-squares matches Python byte-for-byte.
    var inv_norms = alloc[Float32](n)
    for i in range(n):
        var nm = Float32(sqrt(_sumsq_f64(X + i * d, d)))
        norms_out[i] = nm
        inv_norms[i] = Float32(1.0) / nm if nm > Float32(1e-8) else Float32(1.0 / 1e-8)

    # 2. Pre-transpose R so the kernel sees coalesced R_T reads.
    #    R_T[j, k] = R[k, j] stored row-major: R_T[j*d+k] = q.R.data[k*d+j].
    #    O(d²) = ~148K ops at d=384; negligible vs O(n*d²) kernel work.
    var R_T = alloc[Float32](d * d)
    for k in range(d):
        for j in range(d):
            R_T[j * d + k] = q.R.data[k * d + j]

    # 3. Stage to device, launch kernel, drain back.
    var ctx = DeviceContext()

    var X_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var RT_host = ctx.enqueue_create_host_buffer[DType.float32](d * d)
    var inv_host = ctx.enqueue_create_host_buffer[DType.float32](n)
    var bnd_host = ctx.enqueue_create_host_buffer[DType.float32](n_b)
    ctx.synchronize()

    for i in range(n * d):
        X_host[i] = X[i]
    for i in range(d * d):
        RT_host[i] = R_T[i]
    for i in range(n):
        inv_host[i] = inv_norms[i]
    for i in range(n_b):
        bnd_host[i] = q.cb.boundaries[i]

    var X_dev = ctx.enqueue_create_buffer[DType.float32](n * d)
    var RT_dev = ctx.enqueue_create_buffer[DType.float32](d * d)
    var inv_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var bnd_dev = ctx.enqueue_create_buffer[DType.float32](n_b)
    var idx_dev = ctx.enqueue_create_buffer[DType.uint8](n * d)

    ctx.enqueue_copy(dst_buf=X_dev, src_buf=X_host)
    ctx.enqueue_copy(dst_buf=RT_dev, src_buf=RT_host)
    ctx.enqueue_copy(dst_buf=inv_dev, src_buf=inv_host)
    ctx.enqueue_copy(dst_buf=bnd_dev, src_buf=bnd_host)

    comptime BLOCK_X = 16
    comptime BLOCK_Y = 16
    var grid_x = (d + BLOCK_X - 1) // BLOCK_X
    var grid_y = (n + BLOCK_Y - 1) // BLOCK_Y

    ctx.enqueue_function[_encode_kernel, _encode_kernel](
        X_dev.unsafe_ptr(),
        RT_dev.unsafe_ptr(),
        inv_dev.unsafe_ptr(),
        bnd_dev.unsafe_ptr(),
        idx_dev.unsafe_ptr(),
        Int32(n),
        Int32(d),
        Int32(n_b),
        grid_dim=(grid_x, grid_y),
        block_dim=(BLOCK_X, BLOCK_Y),
    )

    var idx_host = ctx.enqueue_create_host_buffer[DType.uint8](n * d)
    ctx.enqueue_copy(dst_buf=idx_host, src_buf=idx_dev)
    ctx.synchronize()

    for i in range(n * d):
        indices_out[i] = idx_host[i]

    R_T.free()
    inv_norms.free()
