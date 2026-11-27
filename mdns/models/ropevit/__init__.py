"""The paper's original RopeViT baseline (kept for comparison only).

The production architecture is ``mdns.models.latticenet``; this package exists so
``registry.build("vit-64x6")`` can reproduce the upstream MDNS model and so old
checkpoints released with the paper remain loadable.
"""

from .vit_rope import RopeVIT, get_rope_vit_model

__all__ = ["RopeVIT", "get_rope_vit_model"]
