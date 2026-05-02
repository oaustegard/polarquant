"""Tiny owning Float32 / Float64 matrices in row-major order.

Just enough to support the encode hot path. Indexing is unchecked; this
is internal scaffolding, not a public collection.

Note on origin: `MutExternalOrigin` marks the heap data as externally
owned from Mojo's borrow-checker perspective; the struct's own `__del__`
is responsible for freeing.
"""

from std.memory import alloc, UnsafePointer


struct Matrix(Movable):
    var data: UnsafePointer[Float32, MutExternalOrigin]
    var rows: Int
    var cols: Int

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = alloc[Float32](rows * cols)
        for i in range(rows * cols):
            self.data[i] = Float32(0.0)

    def __del__(deinit self):
        self.data.free()

    def get(self, i: Int, j: Int) -> Float32:
        return self.data[i * self.cols + j]

    def set(mut self, i: Int, j: Int, v: Float32):
        self.data[i * self.cols + j] = v


struct MatrixF64(Movable):
    var data: UnsafePointer[Float64, MutExternalOrigin]
    var rows: Int
    var cols: Int

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = alloc[Float64](rows * cols)
        for i in range(rows * cols):
            self.data[i] = Float64(0.0)

    def __del__(deinit self):
        self.data.free()

    def get(self, i: Int, j: Int) -> Float64:
        return self.data[i * self.cols + j]

    def set(mut self, i: Int, j: Int, v: Float64):
        self.data[i * self.cols + j] = v

    def to_float32(self) -> Matrix:
        var out = Matrix(self.rows, self.cols)
        for i in range(self.rows * self.cols):
            out.data[i] = Float32(self.data[i])
        return out^
