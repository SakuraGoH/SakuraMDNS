"""MDNS: masked diffusion neural sampler, refactored for XPU/CUDA/CPU.

Subpackages
-----------
device    : device + autocast abstraction (xpu / cuda / cpu)
samplers  : MCMC data generators (C++ CPU kernels, GPU kernels)
"""

__version__ = "0.2.0"

from . import device  # noqa: F401

__all__ = ["device", "__version__"]
