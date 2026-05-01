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
from src.gpu.device import is_gpu_available
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
    print("  polarquant encode <input.npy> --bits N (--seed S | --params P) [--device cpu|gpu] -o <out.pq>")
    print("  polarquant search <index.pq> <query.npy> --k K (--seed S | --params P) [--device cpu|gpu] [--top T]")
    print("                   [--twostage --candidates N --coarse-precision K]")
    print("  polarquant decode <index.pq> (--seed S | --params P) [--precision P] -o <out.npy>")


def _parse_device(args: List[String]) raises -> String:
    """Parse --device flag. Returns 'cpu' (default) or 'gpu'.

    Errors if --device is given an unknown value, or if 'gpu' is requested
    but no GPU is available (the GPU kernels are stubbed pending issue #42,
    so `is_gpu_available()` returns False until they land).
    """
    var idx = _arg_idx(args, String("--device"))
    if idx < 0:
        return String("cpu")
    var dev = _arg_str(args, idx + 1)
    if dev != String("cpu") and dev != String("gpu"):
        raise Error(String("--device must be 'cpu' or 'gpu', got: ") + dev)
    if dev == String("gpu") and not is_gpu_available():
        raise Error(
            "--device gpu requested but no GPU is available. "
            "GPU kernels are not yet implemented (see issue #42). "
            "Drop the flag (or pass --device cpu) to run on CPU."
        )
    return dev


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

    var device = _parse_device(args)

    var rng_idx = _arg_idx(args, String("--rng"))
    var rng_choice = String("numpy")
    if rng_idx >= 0:
        rng_choice = _arg_str(args, rng_idx + 1)
        if rng_choice != "numpy" and rng_choice != "xoshiro":
            raise Error("encode: --rng must be 'numpy' (default) or 'xoshiro'")

    print("encode:", input_path, "→", out_path, "(bits =", bits, ", device =", device, ", rng =", rng_choice, ")")
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

    var device = _parse_device(args)

    # Two-stage mode flags
    var twostage_idx = _arg_idx(args, String("--twostage"))
    var use_twostage = twostage_idx >= 0
    if use_twostage and device == String("gpu"):
        raise Error(
            "search --twostage --device gpu: not supported. "
            "Issue #42 covers adc_search only; two-stage GPU is a follow-up."
        )
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
