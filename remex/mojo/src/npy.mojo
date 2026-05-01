"""Minimal .npy reader for 2D float32, C-contiguous arrays.

Spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

We support v1.0 and v2.0 magic headers, dtype='<f4' (little-endian
float32), fortran_order=False, ndim=2. Anything else raises an error.
"""

from std.pathlib import Path
from std.memory import alloc, UnsafePointer


struct Npy2D(Movable):
    """Owning buffer of a (rows, cols) float32 array loaded from .npy."""
    var data: UnsafePointer[Float32, MutExternalOrigin]
    var rows: Int
    var cols: Int

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = alloc[Float32](rows * cols)

    def __del__(deinit self):
        self.data.free()

    def get(self, i: Int, j: Int) -> Float32:
        return self.data[i * self.cols + j]


def _byte_str(s: String, p: Int) raises -> String:
    """Return s[p] as a 1-char ASCII String."""
    return chr(Int(s.as_bytes()[p]))


def _parse_int(s: String, mut pos: Int) raises -> Int:
    """Parse a non-negative integer starting at `pos` in `s`. Advances pos."""
    var v: Int = 0
    var found: Bool = False
    while pos < len(s):
        var ch = _byte_str(s, pos)
        if ch >= "0" and ch <= "9":
            v = v * 10 + (Int(ord(ch)) - Int(ord("0")))
            pos += 1
            found = True
        else:
            break
    if not found:
        raise Error("expected integer in .npy header")
    return v


def _find_quoted(s: String, key: String) raises -> String:
    """Find a single-quoted string value associated with `'key'`."""
    var marker = String("'") + key + String("'")
    var idx = s.find(marker)
    if idx < 0:
        raise Error(String("missing '") + key + String("' in .npy header"))
    # Skip ahead to ':' then to opening quote
    var p = idx + len(marker)
    while p < len(s) and _byte_str(s, p) != ":":
        p += 1
    p += 1
    while p < len(s) and _byte_str(s, p) != "'":
        p += 1
    if p >= len(s):
        raise Error("malformed .npy header (no opening quote for value)")
    p += 1
    var start = p
    while p < len(s) and _byte_str(s, p) != "'":
        p += 1
    if p >= len(s):
        raise Error("malformed .npy header (no closing quote for value)")
    var out = String("")
    for k in range(start, p):
        out += _byte_str(s, k)
    return out


def _find_bool(s: String, key: String) raises -> Bool:
    var marker = String("'") + key + String("':")
    var idx = s.find(marker)
    if idx < 0:
        raise Error(String("missing '") + key + String("' in .npy header"))
    var p = idx + len(marker)
    while p < len(s) and _byte_str(s, p) == " ":
        p += 1
    # Look for True/False
    if p + 4 <= len(s):
        var four = String("")
        for k in range(p, p + 4):
            four += _byte_str(s, k)
        if four == "True":
            return True
    if p + 5 <= len(s):
        var five = String("")
        for k in range(p, p + 5):
            five += _byte_str(s, k)
        if five == "False":
            return False
    raise Error(String("could not parse '") + key + String("' value"))


def _parse_shape(s: String, mut rows: Int, mut cols: Int) raises:
    """Parse the shape tuple, store rows/cols. Requires 2D."""
    var marker = String("'shape':")
    var idx = s.find(marker)
    if idx < 0:
        raise Error("missing 'shape' in .npy header")
    var p = idx + len(marker)
    while p < len(s) and _byte_str(s, p) != "(":
        p += 1
    if p >= len(s):
        raise Error("malformed shape (no '(')")
    p += 1
    while p < len(s) and _byte_str(s, p) == " ":
        p += 1
    rows = _parse_int(s, p)
    while p < len(s) and _byte_str(s, p) != ",":
        p += 1
    if p >= len(s):
        raise Error("expected 2D array (got 1D shape)")
    p += 1
    while p < len(s) and _byte_str(s, p) == " ":
        p += 1
    if _byte_str(s, p) == ")":
        raise Error("expected 2D array, got 1D")
    cols = _parse_int(s, p)


def load_npy_2d_f32(path: String) raises -> Npy2D:
    """Load a 2D float32 C-contiguous .npy file."""
    var raw = Path(path).read_bytes()
    if len(raw) < 10:
        raise Error("file too small to be .npy")
    # Magic: \x93 N U M P Y
    if (Int(raw[0]) != 0x93 or Int(raw[1]) != Int(ord("N")) or
            Int(raw[2]) != Int(ord("U")) or Int(raw[3]) != Int(ord("M")) or
            Int(raw[4]) != Int(ord("P")) or Int(raw[5]) != Int(ord("Y"))):
        raise Error("bad .npy magic")
    var major = Int(raw[6])
    var header_len: Int
    var data_offset: Int
    if major == 1:
        header_len = Int(raw[8]) | (Int(raw[9]) << 8)
        data_offset = 10 + header_len
    elif major == 2 or major == 3:
        if len(raw) < 12:
            raise Error("file too small for v2.0/3.0 header")
        header_len = (
            Int(raw[8])
            | (Int(raw[9]) << 8)
            | (Int(raw[10]) << 16)
            | (Int(raw[11]) << 24)
        )
        data_offset = 12 + header_len
    else:
        raise Error("unsupported .npy version (need 1.x, 2.x, or 3.x)")

    # Decode header (ASCII)
    var header_offset = data_offset - header_len
    var header = String("")
    for i in range(header_offset, header_offset + header_len):
        header += chr(Int(raw[i]))

    var dtype = _find_quoted(header, "descr")
    if dtype != "<f4" and dtype != "|f4" and dtype != "=f4":
        raise Error(String("only float32 (<f4) supported, got dtype=") + dtype)

    var fortran = _find_bool(header, "fortran_order")
    if fortran:
        raise Error(".npy must be C-contiguous (fortran_order=False)")

    var rows: Int = 0
    var cols: Int = 0
    _parse_shape(header, rows, cols)
    var n = rows * cols
    var expected_bytes = data_offset + n * 4
    if len(raw) < expected_bytes:
        raise Error("truncated .npy data section")

    var out = Npy2D(rows, cols)
    # Copy float32 data little-endian (assume host is little-endian, which
    # is true on x86/ARM64). Reinterpret bytes 4 at a time.
    var bp = out.data.bitcast[UInt8]()
    for i in range(n * 4):
        bp[i] = raw[data_offset + i]
    return out^


def _int_to_str(v: Int) -> String:
    """Convert a non-negative integer to its decimal ASCII string."""
    if v == 0:
        return String("0")
    var digits = String("")
    var x = v
    while x > 0:
        var d = x % 10
        digits = chr(Int(ord("0")) + d) + digits
        x = x // 10
    return digits


def save_npy_2d_f32(path: String,
                    data: UnsafePointer[Float32, MutExternalOrigin],
                    rows: Int, cols: Int) raises:
    """Write a 2D float32 C-contiguous .npy file (v1.0 format).

    The header is padded with spaces so that `(magic + version + len + header)`
    is a multiple of 64 bytes — matches `np.save`'s alignment convention,
    so NumPy can read what we write.
    """
    # Build header dict in NumPy's exact format. The trailing space + newline
    # before the closing brace mimics np.save's header.
    var header_core = (
        String("{'descr': '<f4', 'fortran_order': False, 'shape': (")
        + _int_to_str(rows) + String(", ") + _int_to_str(cols) + String("), }")
    )
    # Pad with spaces so total preamble + header_len is a multiple of 64.
    # Final header byte must be '\n'.
    var preamble = 10  # magic (6) + version (2) + header_len (2)
    var min_total = preamble + len(header_core) + 1  # +1 for '\n'
    var aligned_total = ((min_total + 63) // 64) * 64
    var pad = aligned_total - min_total
    var header_str = header_core
    for _ in range(pad):
        header_str += String(" ")
    header_str += String("\n")
    var header_len = len(header_str)
    var data_offset = preamble + header_len

    var n_floats = rows * cols
    var total = data_offset + n_floats * 4
    var buf = alloc[UInt8](total)
    buf[0] = UInt8(0x93)
    buf[1] = UInt8(ord("N"))
    buf[2] = UInt8(ord("U"))
    buf[3] = UInt8(ord("M"))
    buf[4] = UInt8(ord("P"))
    buf[5] = UInt8(ord("Y"))
    buf[6] = UInt8(1)  # major
    buf[7] = UInt8(0)  # minor
    buf[8] = UInt8(header_len & 0xFF)
    buf[9] = UInt8((header_len >> 8) & 0xFF)

    var hbytes = header_str.as_bytes()
    for i in range(header_len):
        buf[preamble + i] = hbytes[i]

    var src_bytes = data.bitcast[UInt8]()
    for i in range(n_floats * 4):
        buf[data_offset + i] = src_bytes[i]

    var span = Span[UInt8, MutExternalOrigin](ptr=buf, length=total)
    Path(path).write_bytes(span)
    buf.free()
