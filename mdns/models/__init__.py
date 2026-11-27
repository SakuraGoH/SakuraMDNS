"""Model architectures and the architecture registry.

``latticenet``  residual / NBT+Transformer hybrid backbone (new default)
``registry``    named presets: block count, trunk width, bottleneck width,
                compression ratio, transformer width/heads
"""

from . import latticenet, registry
from .latticenet import LatticeNet, NBTBlock, ResBlock
from .registry import DEFAULT_PRESET, build, build_from_config, list_presets

__all__ = ["latticenet", "registry", "LatticeNet", "ResBlock", "NBTBlock",
           "build", "build_from_config", "list_presets", "DEFAULT_PRESET"]
