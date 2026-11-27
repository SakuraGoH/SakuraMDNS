"""Device abstraction so the codebase runs unchanged on CUDA / Intel XPU / CPU.

The upstream MDNS code hard-codes ``'cuda:0'`` and ``torch.amp.autocast('cuda')``.
On this machine torch is built for Intel XPU (``torch.version.cuda is None``), so
those call sites silently fall back to fp32 (autocast disables itself) or crash
(``device='cuda:0'``).  Every new module should go through this helper.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch


# --------------------------------------------------------------------- query
def has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def has_cuda() -> bool:
    return torch.cuda.is_available()


def device_kind(device: torch.device | str | None = None) -> str:
    """'xpu' | 'cuda' | 'cpu'"""
    if device is None:
        return "xpu" if has_xpu() else ("cuda" if has_cuda() else "cpu")
    return torch.device(device).type


def default_device() -> torch.device:
    """Best available accelerator, CPU as last resort."""
    if has_xpu():
        return torch.device("xpu:0")
    if has_cuda():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve(spec: str | torch.device | None) -> torch.device:
    """Map user-facing aliases ('auto', 'gpu', 'cuda', 'xpu', 'cpu') to a device."""
    if spec is None or spec == "auto":
        return default_device()
    if isinstance(spec, torch.device):
        return spec
    s = str(spec).lower()
    if s in ("gpu", "xpu", "xpu:0"):
        return default_device() if not has_xpu() else torch.device("xpu:0")
    if s.startswith("cuda"):
        return torch.device("cuda:0" if not has_cuda() else spec)
    if s.startswith("xpu"):
        return torch.device("xpu:0" if not has_xpu() else spec)
    return torch.device("cpu")


def describe(device: torch.device | None = None) -> str:
    dev = resolve(device)
    kind = dev.type
    if kind == "xpu":
        name = torch.xpu.get_device_name(dev)
        props = torch.xpu.get_device_properties(dev)
        return f"xpu ({name}, {props.max_compute_units} EU, {props.total_memory // (1 << 20)} MB)"
    if kind == "cuda":
        name = torch.cuda.get_device_name(dev)
        total = torch.cuda.get_device_properties(dev).total_memory // (1 << 20)
        return f"cuda ({name}, {total} MB)"
    return f"cpu ({torch.get_num_threads()} threads)"


# ------------------------------------------------------------------ amp -----
def supports_bf16(device: torch.device | str | None = None) -> bool:
    """bf16 is supported on XPU/CUDA and emulated on CPU; probe with a tiny op."""
    dev = resolve(device)
    try:
        a = torch.randn(8, 8, device=dev, dtype=torch.bfloat16)
        (a @ a).float().sum().item()
        return True
    except Exception:
        return False


def amp_dtype(device: torch.device | str | None = None) -> torch.dtype:
    dev = resolve(device)
    if dev.type == "cpu":
        return torch.float32
    return torch.bfloat16 if supports_bf16(dev) else torch.float16


def autocast(device: torch.device | str | None = None, dtype: torch.dtype | None = None,
             enabled: bool = True):
    """Device-correct autocast context (replaces torch.amp.autocast('cuda'))."""
    dev = resolve(device)
    if dev.type == "cpu" or not enabled:
        return nullcontext()
    return torch.amp.autocast(dev.type, dtype=dtype or amp_dtype(dev), enabled=True)


# ---------------------------------------------------------- synchronisation
def synchronize(device: torch.device | str | None = None) -> None:
    dev = resolve(device)
    if dev.type == "xpu":
        torch.xpu.synchronize(dev)
    elif dev.type == "cuda":
        torch.cuda.synchronize(dev)


def empty_cache(device: torch.device | str | None = None) -> None:
    dev = resolve(device)
    if dev.type == "xpu":
        torch.xpu.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def memory_allocated_mb(device: torch.device | str | None = None) -> float:
    dev = resolve(device)
    if dev.type == "xpu":
        return torch.xpu.memory_allocated(dev) / (1 << 20)
    if dev.type == "cuda":
        return torch.cuda.memory_allocated(dev) / (1 << 20)
    return 0.0


# ------------------------------------------------------------------ misc ----
def setup_threads(n: int | None = None) -> int:
    """Pin CPU thread count (used by the C++ samplers and torch CPU ops)."""
    n = n or int(os.environ.get("MDNS_CPU_THREADS", max(1, (os.cpu_count() or 4) - 2)))
    torch.set_num_threads(n)
    return n
