"""MCMC samplers: fast native kernels + GPU implementations.

Public surface
--------------
cpp_ext   : C++ (OpenMP) kernels, the CPU workhorse
gpu       : torch/XPU kernels (checkerboard, Wolff, Swendsen-Wang)
reference : slow pure-python implementations kept for validation only
"""

from . import cpp_ext

__all__ = ["cpp_ext"]
