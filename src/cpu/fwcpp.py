"""Familywise loss on CPU (C++ kernel, sparse backward).  Builds fw_cpu.cpp on first import.

    from fwcpp import familywise_loss
    loss = familywise_loss(logits, target, reduction="mean")   # drop-in for F.cross_entropy (CPU)
    loss.backward()
"""
import os, hashlib, torch
from torch.utils.cpp_extension import load

_here = os.path.dirname(os.path.abspath(__file__))
# FW_CPU_CFLAGS overrides the architecture flags (e.g. "-mavx2 -mfma" or "-mno-avx" to test the fallbacks)
_arch = os.environ.get("FW_CPU_CFLAGS", "-march=native").split()
_ext = load(name="fw_cpu_ext_" + hashlib.md5(" ".join(_arch).encode()).hexdigest()[:8],
            sources=[os.path.join(_here, "fw_cpu.cpp")],
            extra_cflags=["-O3", "-fopenmp", "-fno-math-errno"] + _arch, extra_ldflags=["-fopenmp"], verbose=False)
_RED = {"sum": 0, "mean": 1}

def familywise_loss(logits, target, reduction="mean"):
    return _ext.familywise_loss(logits, target, _RED[reduction])

def fw_loss_grad(Z, y, scale=1.0, flat_kmax=2**62, serial_below=100_000, use_chunked=True):
    """(per-sample loss [N] float64, scale * gradient [N,K], kappa_bar [N]) as torch tensors."""
    return _ext.fw_loss_grad(Z, y, scale, flat_kmax, serial_below, use_chunked)

set_params = _ext.set_params
simd = _ext.simd
