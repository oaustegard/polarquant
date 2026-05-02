"""GPU device discovery for the Mojo encode + ADC search path.

Scaffolding only — the kernels in `encode.mojo` and `adc.mojo` are not yet
implemented. This module exposes a single `is_gpu_available()` predicate
so the CLI, tests, and bench drivers can dispatch / skip cleanly until
the real kernels land. See issue #42.

Implementation will live behind MAX's `DeviceContext`; until that lands,
`is_gpu_available()` is a compile-time `False`. Wire the real probe in
when the kernels arrive — at that point this file becomes the single
place that needs to switch from stub to live, and the rest of the
dispatch (CLI, tests, bench) keeps working unchanged.
"""


fn is_gpu_available() -> Bool:
    """Return True iff a MAX-supported GPU is reachable from this process.

    Stub: always False until the kernel layer is implemented. See issue #42.
    """
    return False
