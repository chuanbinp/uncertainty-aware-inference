"""
nvtx_utils.py
─────────────────────────────────────────────────────────────────────────────
NVTX (NVIDIA Tools Extension) annotations for Nsight Systems capture.

Provides context managers that emit coloured range markers into the nsys
timeline so that:
  - nsys can restrict capture to just the profiling region (not warmup/load)
  - ncu --nvtx-include targets only the generate() steps for kernel replay

Falls back to no-ops silently when NVTX is unavailable (CPU-only machines,
environments without torch.cuda.nvtx).
─────────────────────────────────────────────────────────────────────────────
"""

import contextlib
from typing import Optional

try:
    import torch.cuda.nvtx as _nvtx
    _NVTX_AVAILABLE = True
except ImportError:
    _nvtx = None
    _NVTX_AVAILABLE = False

try:
    import nvtx as _nvtx_pkg
    _NVTX_PKG_AVAILABLE = True
except ImportError:
    _nvtx_pkg = None
    _NVTX_PKG_AVAILABLE = False

NVTX_AVAILABLE: bool = _NVTX_AVAILABLE
DOMAIN = "UAI_PROFILER"


class Color:
    GREEN  = 0x00FF00   # model load
    YELLOW = 0xFFFF00   # warmup steps
    CYAN   = 0x00FFFF   # profiling region (outer) — nsys/ncu capture this
    BLUE   = 0x0000FF   # individual profiling step
    RED    = 0xFF0000   # generate() call inside a step
    WHITE  = 0xFFFFFF   # probe / misc


@contextlib.contextmanager
def NvtxRange(message: str, color: Optional[int] = None, domain: str = DOMAIN):
    """Push an NVTX range on enter, pop on exit. No-op when NVTX unavailable."""
    if not _NVTX_AVAILABLE:
        yield
        return
    if _NVTX_PKG_AVAILABLE and color is not None:
        rng = _nvtx_pkg.start(message, color=color, domain=domain)
        try:
            yield
        finally:
            _nvtx_pkg.end(rng)
    else:
        _nvtx.range_push(message)
        try:
            yield
        finally:
            _nvtx.range_pop()


@contextlib.contextmanager
def model_load_range(config_key: str):
    with NvtxRange(f"model_load/{config_key}", color=Color.GREEN):
        yield


@contextlib.contextmanager
def warmup_range(step: int):
    with NvtxRange(f"warmup/step_{step}", color=Color.YELLOW):
        yield


@contextlib.contextmanager
def profiling_region():
    """
    Outer NVTX range for the entire profiling block.

    nsys restricts capture to this range via:
        nsys profile --capture-range=nvtx --nvtx-capture="profiling_region"
    ncu restricts kernel replay via:
        ncu --nvtx --nvtx-include="profiling_region"
    """
    with NvtxRange("profiling_region", color=Color.CYAN):
        yield


@contextlib.contextmanager
def profile_step_range(step: int):
    with NvtxRange(f"profile_step_{step}", color=Color.BLUE):
        yield


@contextlib.contextmanager
def generate_range(step: int, n_tokens: int):
    with NvtxRange(f"generate/step_{step}/tokens_{n_tokens}", color=Color.RED):
        yield


def probe_nvtx() -> dict:
    """Check NVTX availability. Call at startup before using --nvtx flag."""
    if not _NVTX_AVAILABLE:
        return {
            "available": False,
            "backend": "none",
            "domain_support": False,
            "message": "torch.cuda.nvtx not available. Install CUDA PyTorch or: pip install nvtx",
        }
    backend = "nvtx_pkg" if _NVTX_PKG_AVAILABLE else "torch.cuda.nvtx"
    try:
        with NvtxRange("nvtx_probe_test", color=Color.WHITE):
            import torch
            _ = torch.zeros(1)
        return {
            "available": True,
            "backend": backend,
            "domain_support": _NVTX_PKG_AVAILABLE,
            "message": f"NVTX active ({backend}). Domain '{DOMAIN}' "
                       f"{'supported' if _NVTX_PKG_AVAILABLE else 'not supported — pip install nvtx'}.",
        }
    except Exception as e:
        return {
            "available": False,
            "backend": backend,
            "domain_support": False,
            "message": f"NVTX push/pop failed: {e}",
        }
