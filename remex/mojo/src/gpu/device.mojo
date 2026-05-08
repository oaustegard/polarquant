"""GPU device discovery for the Mojo encode + ADC search path.

Probes whether a Mojo-supported accelerator is present at compile time
via `std.sys.has_accelerator()`. Mojo 1.0 single-source kernels target
NVIDIA, AMD, and Apple Metal from the same source.

Apple Metal is detected by elimination: an accelerator is present and
it is neither NVIDIA nor AMD (i.e. Apple Silicon's integrated GPU,
which is the only other GPU architecture Mojo currently targets). Used
by the per-stage device-routing policy in `polarquant.mojo` because
Apple's TBDR architecture has measured behavior different from
NVIDIA/AMD SIMT GPUs — see `bench/RESULTS.md § Mojo port` and PR #67.
"""

from std.sys import has_accelerator
from std.sys.info import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator


def is_gpu_available() -> Bool:
    """Return True iff a Mojo-supported GPU is reachable from this build."""
    comptime if has_accelerator():
        return True
    else:
        return False


def is_apple_gpu() -> Bool:
    """Return True iff the accelerator is Apple Silicon's integrated GPU.

    True when an accelerator is present that is neither NVIDIA nor AMD;
    in Mojo 1.0 the remaining target is Apple Metal. Used by `polarquant`'s
    `auto` device policy to avoid GPU encode on Apple Metal, which is
    measured 3.7× slower than CPU at d=384 (see PR #67).
    """
    comptime if has_accelerator() \
            and not has_nvidia_gpu_accelerator() \
            and not has_amd_gpu_accelerator():
        return True
    else:
        return False
