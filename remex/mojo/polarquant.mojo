"""polarquant CLI — Mojo port of the remex compress/search workflow.

Usage:
    polarquant encode <input.npy> --bits N [--seed S | --params P.bin] -o <out.pq>
    polarquant search <index.pq> <query.npy> --k K [--seed S | --params P.bin]
                       [--top <int>]

`--seed S`   computes (R, codebook) from seed S in Mojo (not bit-identical to
             a Python Quantizer with the same seed; see README).
`--params P` loads (R, codebook) from a `.params` file written by
             `remex.save_params(quantizer, path)` — bit-identical encoding.

Outputs `index.pq`, the binary container readable by `remex.load_pq()`.
"""

from std.sys import argv
from std.memory import alloc, UnsafePointer
from src.codebook import Codebook, NestedCodebook, lloyd_max_codebook, nested_codebooks_from
from src.matrix import Matrix
from src.npy import load_npy_2d_f32, save_npy_2d_f32
from src.params_format import load_params
from src.pq_format import save_pq, load_pq
from src.quantizer import Quantizer, encode_batch, adc_search, search_twostage, decode_batch
from src.packing import pack, packed_nbytes
from src.rotation import haar_rotation, haar_rotation_numpy
from src.gpu.device import is_apple_gpu, is_gpu_available
from src.gpu.encode import gpu_encode_batch
from src.gpu.adc import gpu_adc_search


def _arg_str(args: List[String], idx: Int) raises -> String:
    if idx >= len(args):
        raise Error("missing argument")
    return args[idx]


def _arg_idx(args: List[String], flag: String) -> Int:
    for i in range(len(args)):
        if args[i] == flag:
            return i
    return -1


def _build_quantizer(d: Int, bits: Int, seed: UInt64,
                     params_path: String,
                     rng_choice: String) raises -> Quantizer:
    """Build a Quantizer from either a .params file or a seed.

    Seed-based modes (default `numpy`) match Python `Quantizer(seed=S)` byte-for-byte.
    `xoshiro` is the legacy fast self-contained Mojo path (no Python parity).
    """
    if len(params_path) > 0:
        var R = Matrix(d, d)
        var cb = Codebook(bits)
        load_params(params_path, R, cb)
        return Quantizer(R^, cb^, d, bits, seed)
    if rng_choice == "xoshiro":
        var R = haar_rotation(d, seed)
        var cb = lloyd_max_codebook(d, bits)
        return Quantizer(R^, cb^, d, bits, seed)
    # default: numpy-compatible
    var R = haar_rotation_numpy(d, seed)
    var cb = lloyd_max_codebook(d, bits)
    return Quantizer(R^, cb^, d, bits, seed)


def _print_usage():
    print("usage:")
    print("  polarquant encode <input.npy> --bits N (--seed S | --params P) [--device auto|cpu|gpu] -o <out.pq>")
    print("  polarquant search <index.pq> <query.npy> --k K (--seed S | --params P) [--device auto|cpu|gpu] [--top T]")
    print("                   [--twostage --candidates N --coarse-precision K]")
    print("  polarquant decode <index.pq> (--seed S | --params P) [--precision P] -o <out.npy>")
    print("")
    print("--device defaults to 'auto', which picks the most efficient backend per stage:")
    print("  encode: always CPU (GPU encode is currently slower on every measured platform;")
    print("          Apple Metal is 3.7x slower at d=384, see bench/RESULTS.md).")
    print("  search: GPU if an accelerator is present, else CPU (GPU search is faster on")
    print("          Apple Metal via the corpus cache; non-Apple GPUs are not yet measured).")
    print("Use --device cpu or --device gpu to force a backend (e.g. for benchmarking).")


def _parse_device(args: List[String]) raises -> String:
    """Parse --device flag. Returns 'auto' (default), 'cpu', or 'gpu'.

    The actual backend used per stage is decided by `_resolve_encode_device`
    and `_resolve_search_device`, which apply per-stage policy when 'auto'
    is requested. Errors only on unknown flag values; 'gpu' availability is
    checked at resolve time so 'auto' can fall back cleanly.
    """
    var idx = _arg_idx(args, String("--device"))
    if idx < 0:
        return String("auto")
    var dev = _arg_str(args, idx + 1)
    if dev != String("auto") and dev != String("cpu") and dev != String("gpu"):
        raise Error(String("--device must be 'auto', 'cpu', or 'gpu', got: ") + dev)
    return dev


def _resolve_encode_device(requested: String) raises -> String:
    """Pick encode backend. Honors explicit cpu/gpu; 'auto' resolves to cpu.

    Auto-policy rationale: GPU encode has been measured slower than CPU on
    every host tested (3.7x slower on Apple M1 at d=384, PR #67). NVIDIA
    and AMD GPU encode are unmeasured; the conservative auto choice is CPU
    until a GPU baseline lands.
    """
    if requested == String("cpu"):
        return String("cpu")
    if requested == String("gpu"):
        if not is_gpu_available():
            raise Error(
                "--device gpu requested but no GPU is available. "
                "Drop the flag (or pass --device cpu / --device auto)."
            )
        if is_apple_gpu():
            print(
                "warning: --device gpu encode on Apple Metal is ~3.7x slower than CPU "
                "(measured at d=384, see bench/RESULTS.md). Proceeding as requested; "
                "use --device auto for the efficient default."
            )
        return String("gpu")
    # auto
    return String("cpu")


def _resolve_search_device(requested: String) raises -> String:
    """Pick search backend. Honors explicit cpu/gpu; 'auto' uses GPU if available.

    Auto-policy rationale: GPU search is measured 1.30x faster than CPU on
    Apple Metal at d=384 (PR #66 corpus cache). The corpus-cache design
    generalizes to NVIDIA/AMD; auto enables it whenever an accelerator is
    reachable, and falls back to CPU otherwise.
    """
    if requested == String("cpu"):
        return String("cpu")
    if requested == String("gpu"):
        if not is_gpu_available():
            raise Error(
                "--device gpu requested but no GPU is available. "
                "Drop the flag (or pass --device cpu / --device auto)."
            )
        return String("gpu")
    # auto
    if is_gpu_available():
        return String("gpu")
    return String("cpu")


def cmd_encode(args: List[String]) raises:
    """Encode subcommand: read .npy, encode, write .pq."""
    if len(args) < 2:
        _print_usage()
        raise Error("encode: missing input.npy")
    var input_path = args[1]

    var bits_idx = _arg_idx(args, String("--bits"))
    if bits_idx < 0:
        raise Error("encode: --bits required")
    var bits = Int(_arg_str(args, bits_idx + 1))

    var seed_idx = _arg_idx(args, String("--seed"))
    var params_idx = _arg_idx(args, String("--params"))
    if seed_idx < 0 and params_idx < 0:
        raise Error("encode: provide --seed or --params")
    var seed: UInt64 = UInt64(42)
    if seed_idx >= 0:
        seed = UInt64(Int(_arg_str(args, seed_idx + 1)))
    var params_path = String("")
    if params_idx >= 0:
        params_path = _arg_str(args, params_idx + 1)

    var out_idx = _arg_idx(args, String("-o"))
    if out_idx < 0:
        raise Error("encode: -o <out.pq> required")
    var out_path = _arg_str(args, out_idx + 1)

    var requested_device = _parse_device(args)
    var device = _resolve_encode_device(requested_device)

    var rng_idx = _arg_idx(args, String("--rng"))
    var rng_choice = String("numpy")
    if rng_idx >= 0:
        rng_choice = _arg_str(args, rng_idx + 1)
        if rng_choice != "numpy" and rng_choice != "xoshiro":
            raise Error("encode: --rng must be 'numpy' (default) or 'xoshiro'")

    var device_note = String("")
    if requested_device == String("auto"):
        device_note = String(" [auto]")
    print("encode:", input_path, "→", out_path, "(bits =", bits, ", device =", device + device_note, ", rng =", rng_choice, ")")
    var X = load_npy_2d_f32(input_path)
    var n = X.rows
    var d = X.cols
    print("  loaded", n, "vectors of dimension", d)

    var q = _build_quantizer(d, bits, seed, params_path, rng_choice)

    # Copy X into a fresh buffer (works around an UnsafePointer borrow oddity
    # observed when passing struct fields across function boundaries).
    var X_buf = alloc[Float32](n * d)
    for i in range(n):
        for j in range(d):
            X_buf[i * d + j] = X.get(i, j)

    var indices = alloc[UInt8](n * d)
    var norms = alloc[Float32](n)
    if device == String("gpu"):
        gpu_encode_batch(q, X_buf, n, indices, norms)
    else:
        encode_batch(q, X_buf, n, indices, norms)
    X_buf.free()

    var nb = packed_nbytes(n * d, bits)
    var packed = alloc[UInt8](nb)
    pack(indices, n * d, bits, packed)
    save_pq(out_path, packed, nb, norms, n, d, bits)
    var total_bytes = nb + n * 4 + 32
    var ratio = Float64(n * d * 4) / Float64(nb + n * 4)
    print("  wrote", total_bytes, "bytes (compression ratio:", ratio, "x)")

    indices.free()
    norms.free()
    packed.free()


def cmd_search(args: List[String]) raises:
    """Search subcommand: load .pq, score query, print top-k."""
    if len(args) < 3:
        _print_usage()
        raise Error("search: missing index.pq or query.npy")
    var index_path = args[1]
    var query_path = args[2]

    var k_idx = _arg_idx(args, String("--k"))
    if k_idx < 0:
        raise Error("search: --k required")
    var k = Int(_arg_str(args, k_idx + 1))

    var seed_idx = _arg_idx(args, String("--seed"))
    var params_idx = _arg_idx(args, String("--params"))
    if seed_idx < 0 and params_idx < 0:
        raise Error("search: provide --seed or --params")
    var seed: UInt64 = UInt64(42)
    if seed_idx >= 0:
        seed = UInt64(Int(_arg_str(args, seed_idx + 1)))
    var params_path = String("")
    if params_idx >= 0:
        params_path = _arg_str(args, params_idx + 1)

    var top_idx = _arg_idx(args, String("--top"))
    var print_top = k
    if top_idx >= 0:
        print_top = Int(_arg_str(args, top_idx + 1))

    var requested_device = _parse_device(args)

    # Two-stage mode flags
    var twostage_idx = _arg_idx(args, String("--twostage"))
    var use_twostage = twostage_idx >= 0
    # Twostage runs on CPU only — force CPU when 'auto'; reject explicit 'gpu'.
    var device: String
    if use_twostage:
        if requested_device == String("gpu"):
            raise Error(
                "search --twostage --device gpu: not supported. "
                "Issue #42 covers adc_search only; two-stage GPU is a follow-up. "
                "Drop the flag (or pass --device auto / --device cpu)."
            )
        device = String("cpu")
    else:
        device = _resolve_search_device(requested_device)
    if requested_device == String("auto"):
        print("search: device resolved to", device, "[auto]")
    var candidates = 0
    var coarse_precision = 0
    if use_twostage:
        var cand_idx = _arg_idx(args, String("--candidates"))
        if cand_idx < 0:
            raise Error("search --twostage: --candidates N required")
        candidates = Int(_arg_str(args, cand_idx + 1))
        var cp_idx = _arg_idx(args, String("--coarse-precision"))
        if cp_idx < 0:
            raise Error("search --twostage: --coarse-precision K required")
        coarse_precision = Int(_arg_str(args, cp_idx + 1))

    var pq = load_pq(index_path)
    var Q = load_npy_2d_f32(query_path)
    if Q.rows != 1:
        raise Error("search: query.npy must be a single (1, d) vector")
    if Q.cols != pq.d:
        raise Error("search: query d mismatch")

    var rng_idx2 = _arg_idx(args, String("--rng"))
    var rng_choice2 = String("numpy")
    if rng_idx2 >= 0:
        rng_choice2 = _arg_str(args, rng_idx2 + 1)
        if rng_choice2 != "numpy" and rng_choice2 != "xoshiro":
            raise Error("search: --rng must be 'numpy' (default) or 'xoshiro'")

    var q_quant = _build_quantizer(pq.d, pq.bits, seed, params_path, rng_choice2)

    # Unpack indices into uint8 (n*d). Copy norms into a fresh buffer too —
    # see test_encode.mojo for the same workaround.
    from src.packing import unpack
    var indices = alloc[UInt8](pq.n * pq.d)
    unpack(pq.packed_indices, pq.n * pq.d, pq.bits, indices)
    var norms_local = alloc[Float32](pq.n)
    for i in range(pq.n):
        norms_local[i] = pq.norms[i]

    # Copy query
    var qbuf = alloc[Float32](pq.d)
    for j in range(pq.d):
        qbuf[j] = Q.get(0, j)

    var top_idx_buf = alloc[Int](k)
    var top_scores = alloc[Float32](k)
    if use_twostage:
        # Build nested centroid tables from the loaded full-precision codebook.
        # Reaching into q_quant.cb is fine — Quantizer is owned locally here.
        var nested = nested_codebooks_from(q_quant.cb, pq.d)
        search_twostage(q_quant, nested, indices, norms_local, pq.n,
                        qbuf, k, candidates, coarse_precision,
                        top_idx_buf, top_scores)
    elif device == String("gpu"):
        gpu_adc_search(q_quant, indices, norms_local, pq.n, qbuf, k, top_idx_buf, top_scores)
    else:
        adc_search(q_quant, indices, norms_local, pq.n, qbuf, k, top_idx_buf, top_scores)

    print("top", print_top, "results:")
    var to_print = print_top if print_top <= k else k
    for i in range(to_print):
        print("  rank", i, ": idx =", top_idx_buf[i], " score =", top_scores[i])

    qbuf.free()
    indices.free()
    norms_local.free()
    top_idx_buf.free()
    top_scores.free()


def cmd_decode(args: List[String]) raises:
    """Decode subcommand: load .pq, reconstruct float32 vectors, write .npy.

    End-to-end reconstruction parity with `Quantizer.decode` from the
    Python library: same params + same .pq + same `--precision` produce
    a float32 (n, d) array matching to within 1e-5 max abs diff.
    """
    if len(args) < 2:
        _print_usage()
        raise Error("decode: missing index.pq")
    var index_path = args[1]

    var seed_idx = _arg_idx(args, String("--seed"))
    var params_idx = _arg_idx(args, String("--params"))
    if seed_idx < 0 and params_idx < 0:
        raise Error("decode: provide --seed or --params")
    var seed: UInt64 = UInt64(42)
    if seed_idx >= 0:
        seed = UInt64(Int(_arg_str(args, seed_idx + 1)))
    var params_path = String("")
    if params_idx >= 0:
        params_path = _arg_str(args, params_idx + 1)

    var prec_idx = _arg_idx(args, String("--precision"))
    var precision = 0  # 0 means full precision
    if prec_idx >= 0:
        precision = Int(_arg_str(args, prec_idx + 1))

    var rng_idx = _arg_idx(args, String("--rng"))
    var rng_choice = String("numpy")
    if rng_idx >= 0:
        rng_choice = _arg_str(args, rng_idx + 1)
        if rng_choice != "numpy" and rng_choice != "xoshiro":
            raise Error("decode: --rng must be 'numpy' (default) or 'xoshiro'")

    var out_idx = _arg_idx(args, String("-o"))
    if out_idx < 0:
        raise Error("decode: -o <out.npy> required")
    var out_path = _arg_str(args, out_idx + 1)

    var pq = load_pq(index_path)
    print("decode:", index_path, "→", out_path,
          "(n =", pq.n, ", d =", pq.d, ", bits =", pq.bits,
          ", precision =", precision if precision > 0 else pq.bits, ")")

    var q_quant = _build_quantizer(pq.d, pq.bits, seed, params_path, rng_choice)

    # Build nested centroid tables from the loaded codebook so they match
    # what Python would compute for the same Quantizer (mirrors the
    # search_twostage path).
    var nested = nested_codebooks_from(q_quant.cb, pq.d)

    # Unpack indices into uint8 (n*d). Copy norms into a fresh buffer too
    # (UnsafePointer field aliasing workaround — see test_encode.mojo).
    from src.packing import unpack
    var indices = alloc[UInt8](pq.n * pq.d)
    unpack(pq.packed_indices, pq.n * pq.d, pq.bits, indices)
    var norms_local = alloc[Float32](pq.n)
    for i in range(pq.n):
        norms_local[i] = pq.norms[i]

    var X_hat = alloc[Float32](pq.n * pq.d)
    decode_batch(q_quant, nested, indices, norms_local, pq.n,
                 precision, X_hat)

    save_npy_2d_f32(out_path, X_hat, pq.n, pq.d)
    print("  wrote", pq.n * pq.d * 4, "float32 bytes")

    indices.free()
    norms_local.free()
    X_hat.free()


def main() raises:
    var args = argv()
    if len(args) < 2:
        _print_usage()
        return
    var cmd = String(args[1])
    # Slice argv from position 1 forward as the subcommand's own argv.
    var sub_args = List[String]()
    for i in range(1, len(args)):
        sub_args.append(String(args[i]))
    if cmd == "encode":
        cmd_encode(sub_args)
    elif cmd == "search":
        cmd_search(sub_args)
    elif cmd == "decode":
        cmd_decode(sub_args)
    else:
        _print_usage()
        raise Error(String("unknown subcommand: ") + cmd)
