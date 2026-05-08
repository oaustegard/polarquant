"""GPU device discovery for the Mojo encode + ADC search path.

Probes whether a Mojo-supported accelerator is present at compile time
via `std.sys.has_accelerator()`. Mojo 1.0 single-source kernels target
NVIDIA, AMD, and Apple Metal from the same source.
"""

from std.sys import has_accelerator


def is_gpu_available() -> Bool:
    """Return True iff a Mojo-supported GPU is reachable from this build."""
    comptime if has_accelerator():
        return True
    else:
        return False
