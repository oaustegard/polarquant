"""Deterministic PRNG and standard-normal sampling.

Uses splitmix64 to seed xoshiro256++ for uniforms, and the Marsaglia
polar method for standard normals. The same seed yields the same stream
across runs and platforms.

Seed semantics are NOT bit-identical to NumPy's `default_rng(seed)`. To
get an encoding bit-identical to Python remex, dump (R, boundaries) from
Python with `remex.dump_quantizer_params()` and load them in Mojo via
`pq_format.load_params()`. The Mojo CLI exposes both paths.
"""

from std.math import sqrt, log


@fieldwise_init
struct SplitMix64(Copyable, Movable):
    var state: UInt64

    def next(mut self) -> UInt64:
        self.state = self.state + 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)


struct Xoshiro256pp(Copyable, Movable):
    var s0: UInt64
    var s1: UInt64
    var s2: UInt64
    var s3: UInt64
    var _has_spare: Bool
    var _spare: Float64

    def __init__(out self, seed: UInt64):
        var sm = SplitMix64(seed)
        self.s0 = sm.next()
        self.s1 = sm.next()
        self.s2 = sm.next()
        self.s3 = sm.next()
        self._has_spare = False
        self._spare = 0.0

    def _rotl(self, x: UInt64, k: Int) -> UInt64:
        var ku = UInt64(k)
        return (x << ku) | (x >> (UInt64(64) - ku))

    def next_u64(mut self) -> UInt64:
        var result = self._rotl(self.s0 + self.s3, 23) + self.s0
        var t = self.s1 << 17
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= t
        self.s3 = self._rotl(self.s3, 45)
        return result

    def next_uniform(mut self) -> Float64:
        # 53-bit mantissa uniform in [0, 1)
        var bits = self.next_u64() >> 11
        return Float64(bits) * (1.0 / Float64(1 << 53))

    def next_normal(mut self) -> Float64:
        """Marsaglia polar method — produces pairs; caches the spare."""
        if self._has_spare:
            self._has_spare = False
            return self._spare

        var u: Float64 = 0.0
        var v: Float64 = 0.0
        var s: Float64 = 2.0
        while s >= 1.0 or s == 0.0:
            u = 2.0 * self.next_uniform() - 1.0
            v = 2.0 * self.next_uniform() - 1.0
            s = u * u + v * v
        var factor = sqrt(-2.0 * log(s) / s)
        self._spare = v * factor
        self._has_spare = True
        return u * factor
