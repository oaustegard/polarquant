"""GPU ADC search kernel for `polarquant search --device gpu`.

Single-source kernel that runs on Apple Metal, NVIDIA, and AMD via
Mojo 1.0's GPU dispatch. Mirrors `src/quantizer.mojo::adc_search`.

## Strategy

`adc_search` does four things:
  1. q_rot = R @ query              — (d, d) × (d,) matvec
  2. table[j, c] = q_rot[j] * centroids[c]  — (d, n_levels) outer product
  3. score[i] = sum_j table[j, indices[i, j]] * norms[i]   — gather/reduce
  4. top-k selection by score

Steps 1, 2, 4 run on CPU. They're all O(d²) or O(n_levels) or O(n*k) —
small relative to step 3, which is O(n*d) and bandwidth-dominated. Doing
them on CPU keeps the GPU kernel focused and matches the CPU reduction
order exactly, which is needed for the rtol=1e-5 score parity check.

Step 3 runs on GPU: one thread per row, loop j=0..d, gather
`table[j*n_levels + c]` indexed by `indices[row*d+j]`, accumulate
left-to-right, multiply by `norms[row]`. Coalesced reads on `indices`
(adjacent threads → adjacent rows → same j → adjacent indices). Reads
on `table` are uncoalesced gathers — each thread picks a different `c`.
For first cut this lives in global memory; a future pass can stage
`table` to shared (d * n_levels * 4 = ~24 KB at d=384, n_levels=16,
fits comfortably in M1's 32 KB/block).
"""

from std.memory import alloc, UnsafePointer
from std.gpu import block_idx, block_dim, thread_idx
from std.gpu.host import DeviceContext

from src.quantizer import Quantizer, _dot_f32


def _score_kernel(
    indices: UnsafePointer[UInt8, MutAnyOrigin],
    norms: UnsafePointer[Float32, MutAnyOrigin],
    table: UnsafePointer[Float32, MutAnyOrigin],
    scores_out: UnsafePointer[Float32, MutAnyOrigin],
    n: Int32,
    d: Int32,
    n_levels: Int32,
):
    var row = Int32(block_idx.x * block_dim.x + thread_idx.x)
    if row >= n:
        return

    var ri = Int(row)
    var di = Int(d)
    var nli = Int(n_levels)

    # Left-to-right gather + reduce — matches CPU FP order for parity.
    var s: Float32 = 0.0
    var base = ri * di
    for j in range(di):
        var c = Int(indices[base + j])
        s += table[j * nli + c]
    scores_out[ri] = s * norms[ri]


def gpu_adc_search(q: Quantizer,
                   indices: UnsafePointer[UInt8, MutExternalOrigin],
                   norms: UnsafePointer[Float32, MutExternalOrigin],
                   n: Int,
                   query: UnsafePointer[Float32, MutExternalOrigin],
                   k: Int,
                   mut top_idx: UnsafePointer[Int, MutExternalOrigin],
                   mut top_scores: UnsafePointer[Float32, MutExternalOrigin]) raises:
    """GPU mirror of `adc_search`. Same contract as the CPU path."""
    var d = q.d
    var n_levels = q.cb.n_levels

    # 1. q_rot = R @ query  — CPU, matches `_dot_f32` order.
    var q_rot = alloc[Float32](d)
    for i in range(d):
        q_rot[i] = _dot_f32(q.R.data + i * d, query, d)

    # 2. table[j, c] = q_rot[j] * centroids[c]  — CPU.
    var table = alloc[Float32](d * n_levels)
    for j in range(d):
        var qj = q_rot[j]
        var trow = j * n_levels
        for c in range(n_levels):
            table[trow + c] = qj * q.cb.centroids[c]

    # 3. GPU per-row score: gather indices into `table`, accumulate, scale by norm.
    var ctx = DeviceContext()
    var idx_host = ctx.enqueue_create_host_buffer[DType.uint8](n * d)
    var nrm_host = ctx.enqueue_create_host_buffer[DType.float32](n)
    var tbl_host = ctx.enqueue_create_host_buffer[DType.float32](d * n_levels)
    ctx.synchronize()

    for i in range(n * d):
        idx_host[i] = indices[i]
    for i in range(n):
        nrm_host[i] = norms[i]
    for i in range(d * n_levels):
        tbl_host[i] = table[i]

    var idx_dev = ctx.enqueue_create_buffer[DType.uint8](n * d)
    var nrm_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var tbl_dev = ctx.enqueue_create_buffer[DType.float32](d * n_levels)
    var scr_dev = ctx.enqueue_create_buffer[DType.float32](n)

    ctx.enqueue_copy(dst_buf=idx_dev, src_buf=idx_host)
    ctx.enqueue_copy(dst_buf=nrm_dev, src_buf=nrm_host)
    ctx.enqueue_copy(dst_buf=tbl_dev, src_buf=tbl_host)

    comptime BLOCK = 256
    var grid = (n + BLOCK - 1) // BLOCK

    ctx.enqueue_function[_score_kernel, _score_kernel](
        idx_dev.unsafe_ptr(),
        nrm_dev.unsafe_ptr(),
        tbl_dev.unsafe_ptr(),
        scr_dev.unsafe_ptr(),
        Int32(n),
        Int32(d),
        Int32(n_levels),
        grid_dim=grid,
        block_dim=BLOCK,
    )

    var scr_host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=scr_host, src_buf=scr_dev)
    ctx.synchronize()

    var scores = alloc[Float32](n)
    for i in range(n):
        scores[i] = scr_host[i]

    # 4. Top-k on CPU — O(n*k), k typically small.
    var used = alloc[UInt8](n)
    for i in range(n):
        used[i] = UInt8(0)
    var kk = k if k <= n else n
    for outer in range(kk):
        var best_i: Int = -1
        var best_s: Float32 = Float32(0.0)
        for i in range(n):
            if used[i] == UInt8(0):
                if best_i < 0 or scores[i] > best_s:
                    best_i = i
                    best_s = scores[i]
        top_idx[outer] = best_i
        top_scores[outer] = best_s
        used[best_i] = UInt8(1)

    used.free()
    scores.free()
    table.free()
    q_rot.free()
