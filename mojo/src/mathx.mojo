"""Math helpers: standard normal CDF / PDF on N(0, sigma^2).

The Lloyd-Max codebook iteration over N(0, 1/d) uses these. We implement
them with the stdlib `erf` and `exp` so the Mojo binary stays pure with no
SciPy/Python interop at runtime.
"""

from std.math import erf, exp, sqrt, pi


comptime SQRT_2 = 1.4142135623730951
comptime INV_SQRT_2PI = 0.3989422804014327  # 1 / sqrt(2*pi)


def normal_cdf(x: Float64, sigma: Float64) -> Float64:
    """CDF of N(0, sigma^2) at x."""
    return 0.5 * (1.0 + erf(x / (sigma * SQRT_2)))


def normal_pdf(x: Float64, sigma: Float64) -> Float64:
    """PDF of N(0, sigma^2) at x."""
    var z = x / sigma
    return INV_SQRT_2PI / sigma * exp(-0.5 * z * z)


def normal_truncated_mean(a: Float64, b: Float64, sigma: Float64) -> Float64:
    """E[X | a < X <= b] for X ~ N(0, sigma^2).

    Equals -sigma^2 * (pdf(b) - pdf(a)) / (cdf(b) - cdf(a)).
    Used inside Lloyd-Max to recompute centroids from boundaries.
    """
    var pa = normal_pdf(a, sigma)
    var pb = normal_pdf(b, sigma)
    var ca = normal_cdf(a, sigma)
    var cb = normal_cdf(b, sigma)
    var denom = cb - ca
    if denom < 1e-300:
        return 0.5 * (a + b)
    return -(sigma * sigma) * (pb - pa) / denom
