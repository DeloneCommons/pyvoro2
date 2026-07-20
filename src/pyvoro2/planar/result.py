"""Compatibility name for the common structured tessellation result."""

from __future__ import annotations

from ..result import TessellationResult


# Compatibility-only historical name. This must remain an identity alias so
# package and direct-module imports agree with the common result contract.
PlanarComputeResult = TessellationResult


__all__ = ['PlanarComputeResult']
