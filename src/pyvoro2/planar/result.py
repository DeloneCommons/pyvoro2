"""Structured result objects for wrapper-level planar convenience APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .diagnostics import TessellationDiagnostics
from .normalize import NormalizedTopology, NormalizedVertices


@dataclass(frozen=True, slots=True)
class PlanarComputeResult:
    """Structured return value for :func:`pyvoro2.planar.compute`.

    Attributes:
        cells: Raw planar cell dictionaries returned by the compute wrapper
            after any temporary internal geometry has been stripped according to
            the caller's requested output flags.
        tessellation_diagnostics: Wrapper-computed tessellation diagnostics, if
            requested directly or needed for ``tessellation_check``.
        normalized_vertices: Vertex-normalized planar output, if requested via
            ``normalize='vertices'`` or ``normalize='topology'``.
        normalized_topology: Edge-normalized planar topology, if requested via
            ``normalize='topology'``.

    The normalized structures intentionally carry their own augmented cell
    copies. They are not aliases of ``cells`` and may therefore still contain
    geometry that was omitted from the raw wrapper output.
    """

    cells: list[dict[str, Any]]
    tessellation_diagnostics: TessellationDiagnostics | None = None
    normalized_vertices: NormalizedVertices | None = None
    normalized_topology: NormalizedTopology | None = None

    @property
    def has_tessellation_diagnostics(self) -> bool:
        """Whether tessellation diagnostics are present."""

        return self.tessellation_diagnostics is not None

    @property
    def has_normalized_vertices(self) -> bool:
        """Whether vertex normalization output is present."""

        return self.normalized_vertices is not None

    @property
    def has_normalized_topology(self) -> bool:
        """Whether topology normalization output is present."""

        return self.normalized_topology is not None

    @property
    def global_vertices(self) -> np.ndarray | None:
        """Global planar vertices from the available normalized output."""

        if self.normalized_topology is not None:
            return self.normalized_topology.global_vertices
        if self.normalized_vertices is not None:
            return self.normalized_vertices.global_vertices
        return None

    @property
    def global_edges(self) -> list[dict[str, Any]] | None:
        """Global planar edges if topology normalization is available."""

        if self.normalized_topology is None:
            return None
        return self.normalized_topology.global_edges

    def require_tessellation_diagnostics(self) -> TessellationDiagnostics:
        """Return tessellation diagnostics or raise a helpful error."""

        if self.tessellation_diagnostics is None:
            raise ValueError(
                'tessellation diagnostics are not available; pass '
                'return_diagnostics=True or enable tessellation_check'
            )
        return self.tessellation_diagnostics

    def require_normalized_vertices(self) -> NormalizedVertices:
        """Return vertex normalization output or raise a helpful error."""

        if self.normalized_vertices is None:
            raise ValueError(
                'normalized vertices are not available; pass '
                "normalize='vertices' or normalize='topology'"
            )
        return self.normalized_vertices

    def require_normalized_topology(self) -> NormalizedTopology:
        """Return topology normalization output or raise a helpful error."""

        if self.normalized_topology is None:
            raise ValueError(
                'normalized topology is not available; pass '
                "normalize='topology'"
            )
        return self.normalized_topology
