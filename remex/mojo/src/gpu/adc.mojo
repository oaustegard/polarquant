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

## Persistent corpus: GPUCorpus

`gpu_adc_search` (legacy) restages the full corpus (indices + norms) on
every call. At n=10000, d=384 that is ~3.87 MB of H2D per query — the
dominant cost at 68 GB/s shared memory bandwidth.

`GPUCorpus` holds device-resident buffers for indices and norms.  Build
it once from the host arrays and reuse it across many queries via
`gpu_adc_search_corpus`, which only H2D-transfers the per-query lookup
table (~24 KB at d=384, n_levels=16) instead of the full corpus.

Expected transfer reduction: 3.87 MB → 24 KB per query (≈160× less H2D).

Usage::

    var corpus = GPUCorpus(indices, norms, n, d)        # one-time H2D
    for each query:
        gpu_adc_search_corpus(q, corpus, query, k, ...)  # 24 KB H2D

Type note: `DeviceBuffer[DType]` is assumed to come from `std.gpu.host`.
If the actual Mojo 1.0 type name differs, adjust the import accordingly.
"""

from std.memory import alloc, UnsafePointer
from std.gpu import block_idx, block_dim, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

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


def _topk_cpu(scores: UnsafePointer[Float32, MutExternalOrigin],
              n: Int,
              k: Int,
              mut top_idx: UnsafePointer[Int, MutExternalOrigin],
              mut top_scores: UnsafePointer[Float32, MutExternalOrigin]):
    """O(n*k) top-k selection — k is small in practice."""
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


struct GPUCorpus(Movable):
    """Device-resident corpus: indices and norms staged once, reused per query.

    Build once from host buffers; call `gpu_adc_search_corpus` for each
    query. The DeviceContext and device buffers are owned by this struct.

    Note on DeviceBuffer typing: the field types assume `DeviceBuffer[DType]`
    is exported from `std.gpu.host`. Adjust import if the actual release
    uses a different name.
    """
    var ctx: DeviceContext
    var idx_dev: DeviceBuffer[DType.uint8]
    var nrm_dev: DeviceBuffer[DType.float32]
    var n: Int
    var d: Int
    var n_levels: Int

    def __init__(out self,
                 indices: UnsafePointer[UInt8, MutExternalOrigin],
                 norms: UnsafePointer[Float32, MutExternalOrigin],
                 n: Int,
                 d: Int,
                 n_levels: Int) raises:
        self.n = n
        self.d = d
        self.n_levels = n_levels
        self.ctx = DeviceContext()

        # Stage indices (n*d bytes) to device — one-time cost.
        var idx_host = self.ctx.enqueue_create_host_buffer[DType.uint8](n * d)
        self.ctx.synchronize()
        for i in range(n * d):
            idx_host[i] = indices[i]
        self.idx_dev = self.ctx.enqueue_create_buffer[DType.uint8](n * d)
        self.ctx.enqueue_copy(dst_buf=self.idx_dev, src_buf=idx_host)

        # Stage norms (n * 4 bytes) to device — one-time cost.
        var nrm_host = self.ctx.enqueue_create_host_buffer[DType.float32](n)
        self.ctx.synchronize()
        for i in range(n):
            nrm_host[i] = norms[i]
        self.nrm_dev = self.ctx.enqueue_create_buffer[DType.float32](n)
        self.ctx.enqueue_copy(dst_buf=self.nrm_dev, src_buf=nrm_host)

        self.ctx.synchronize()


def gpu_adc_search_corpus(q: Quantizer,
                          corpus: GPUCorpus,
                          query: UnsafePointer[Float32, MutExternalOrigin],
                          k: Int,
                          mut top_idx: UnsafePointer[Int, MutExternalOrigin],
                          mut top_scores: UnsafePointer[Float32, MutExternalOrigin]) raises:
    """ADC search using a pre-staged device corpus.

    Per-query H2D is limited to the lookup table (~24 KB at d=384,
    n_levels=16) instead of the full corpus (~3.87 MB).  The device-side
    indices and norms from `corpus` are reused directly.

    Contract is identical to `gpu_adc_search` except indices + norms are
    already on device.
    """
    var d = q.d
    var n_levels = q.cb.n_levels
    var n = corpus.n

    # 1. q_rot = R @ query  — CPU, matches `_dot_f32` order.
    var q_rot = alloc[Float32](d)
    for i in range(d):
        q_rot[i] = _dot_f32(q.R.data + i * d, query, d)

    # 2. table[j, c] = q_rot[j] * centroids[c]  — CPU, d * n_levels floats.
    var table = alloc[Float32](d * n_levels)
    for j in range(d):
        var qj = q_rot[j]
        var trow = j * n_levels
        for c in range(n_levels):
            table[trow + c] = qj * q.cb.centroids[c]

    # 3. H2D: only the table (~24 KB) — corpus is already on device.
    var tbl_host = corpus.ctx.enqueue_create_host_buffer[DType.float32](d * n_levels)
    corpus.ctx.synchronize()
    for i in range(d * n_levels):
        tbl_host[i] = table[i]
    var tbl_dev = corpus.ctx.enqueue_create_buffer[DType.float32](d * n_levels)
    corpus.ctx.enqueue_copy(dst_buf=tbl_dev, src_buf=tbl_host)

    # 4. GPU per-row score kernel.
    var scr_dev = corpus.ctx.enqueue_create_buffer[DType.float32](n)

    comptime BLOCK = 256
    var grid = (n + BLOCK - 1) // BLOCK

    corpus.ctx.enqueue_function[_score_kernel, _score_kernel](
        corpus.idx_dev.unsafe_ptr(),
        corpus.nrm_dev.unsafe_ptr(),
        tbl_dev.unsafe_ptr(),
        scr_dev.unsafe_ptr(),
        Int32(n),
        Int32(d),
        Int32(n_levels),
        grid_dim=grid,
        block_dim=BLOCK,
    )

    # 5. D2H scores.
    var scr_host = corpus.ctx.enqueue_create_host_buffer[DType.float32](n)
    corpus.ctx.enqueue_copy(dst_buf=scr_host, src_buf=scr_dev)
    corpus.ctx.synchronize()

    var scores = alloc[Float32](n)
    for i in range(n):
        scores[i] = scr_host[i]

    # 6. Top-k on CPU.
    _topk_cpu(scores, n, k, top_idx, top_scores)

    scores.free()
    table.free()
    q_rot.free()


def gpu_adc_search(q: Quantizer,
                   indices: UnsafePointer[UInt8, MutExternalOrigin],
                   norms: UnsafePointer[Float32, MutExternalOrigin],
                   n: Int,
                   query: UnsafePointer[Float32, MutExternalOrigin],
                   k: Int,
                   mut top_idx: UnsafePointer[Int, MutExternalOrigin],
                   mut top_scores: UnsafePointer[Float32, MutExternalOrigin]) raises:
    """GPU mirror of `adc_search`. Same contract as the CPU path.

    Stages the full corpus (indices + norms) to device on every call.
    For repeated queries against the same corpus, prefer `GPUCorpus` +
    `gpu_adc_search_corpus` to eliminate the per-query H2D staging cost.
    """
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
    _topk_cpu(scores, n, k, top_idx, top_scores)

    scores.free()
    table.free()
    q_rot.free()
