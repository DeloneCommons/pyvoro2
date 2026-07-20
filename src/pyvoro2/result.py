"""Common structured result contract for 2D and 3D tessellations."""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import index
from typing import Any, Literal, Sequence

import numpy as np

from ._power_input import ResolvedPowerInput
from ._weight_transforms import weights_to_radii


def _readonly_float_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    nonnegative: bool = False,
) -> np.ndarray:
    """Return a validated, owned, read-only float64 copy."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite values')
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f'{name} must be non-negative')
    owned = np.array(array, dtype=np.float64, copy=True, order='C')
    owned.setflags(write=False)
    return owned


def _readonly_ids(values: np.ndarray, *, n: int) -> np.ndarray:
    """Return a validated, owned, read-only int64 ID copy."""

    array = np.asarray(values)
    if array.shape != (n,):
        raise ValueError(f'ids must have shape ({n},)')
    if n == 0:
        owned = np.empty((0,), dtype=np.int64)
        owned.setflags(write=False)
        return owned
    if array.dtype.kind not in 'iu':
        raise ValueError('ids must contain integers')
    try:
        owned = np.array(array, dtype=np.int64, copy=True, order='C')
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError('ids must contain int64-compatible integers') from exc
    if np.any(owned < 0):
        raise ValueError('ids must be non-negative')
    if np.unique(owned).size != n:
        raise ValueError('ids must be unique')
    owned.setflags(write=False)
    return owned


def _readonly_mask(values: np.ndarray, *, n: int) -> np.ndarray:
    """Return a validated, owned, read-only boolean copy."""

    array = np.asarray(values)
    if array.shape != (n,):
        raise ValueError(f'empty_mask must have shape ({n},)')
    if array.dtype.kind != 'b':
        raise ValueError('empty_mask must contain booleans')
    owned = np.array(array, dtype=np.bool_, copy=True, order='C')
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, slots=True, eq=False)
class TessellationResult:
    """Dimension-neutral structured tessellation data.

    ``TessellationResult`` records common scientific output without erasing
    dimension-specific geometry. It is the default return from both public
    ``compute`` functions; callers that need historical raw records can select
    ``output='cells'`` explicitly.

    Attributes:
        dimension: Explicit spatial dimension, either ``2`` or ``3``.
        domain: Validated domain object used by the computation.
        mode: ``"standard"`` for a Voronoi diagram or ``"power"`` for a
            power/Laguerre diagram.
        sites: Read-only ``(n, dimension)`` copy of validated input coordinates
            in original input order.
        ids: Read-only ``(n,)`` integer IDs in original input order. Omitted
            external IDs are represented by ``0..n-1``.
        cells: Raw user-visible cell dictionaries after external-ID remapping.
        cell_measures: Read-only ``(n,)`` snapshot of input-aligned areas in 2D
            or volumes in 3D. Hidden cells have measure zero.
        empty_mask: Read-only input-aligned boolean snapshot. It includes
            empty cells even when their raw records were omitted.
        input_weights: Read-only copy of mathematical power weights when the
            caller supplied ``weights=``; otherwise ``None``.
        backend_radii: Read-only copy of the exact non-negative radii supplied
            to the native power backend; for weight-first input these equal
            ``sqrt(input_weights + representation_shift)``. ``None`` in
            standard mode.
        representation_shift: Common additive weight shift used to obtain the
            backend radii, or ``None`` for standard mode and direct radii.
        tessellation_diagnostics: Dimension-specific tessellation diagnostics,
            when computed.
        normalized_vertices: Dimension-specific normalized-vertex output, when
            computed.
        normalized_topology: Dimension-specific normalized-topology output,
            when computed.

    The outer object is frozen and every aligned numeric array is an owned,
    read-only copy. ``cells`` is deliberately different: the exact supplied
    list is retained, and its dictionaries and nested geometry remain mutable.
    The measure and empty-mask arrays are construction-time snapshots and do
    not follow later raw-record mutation. Mutating boundary records does affect
    later boundary access and may make :meth:`require_boundaries` raise an
    inconsistency error. Normalization and diagnostic objects retain their own
    dimension-specific mutability contracts.

    Direct construction is supported. The constructor validates raw-cell IDs,
    measures, empty state, representation metadata, and capability metadata
    against the aligned fields rather than repairing inconsistent input. The
    private keyword-only ``_boundaries_available`` and
    ``_periodic_shifts_available`` parameters carry construction capability
    state; the shared builder supplies them for normal integration use. Deep
    copies and pickle round trips preserve the existing snapshot state while
    restoring read-only owned arrays and capability state, even after allowed
    raw-record mutation.
    """

    dimension: Literal[2, 3]
    domain: object
    mode: Literal['standard', 'power']
    sites: np.ndarray
    ids: np.ndarray
    cells: list[dict[str, Any]]
    cell_measures: np.ndarray
    empty_mask: np.ndarray
    input_weights: np.ndarray | None = None
    backend_radii: np.ndarray | None = None
    representation_shift: float | None = None
    tessellation_diagnostics: object | None = None
    normalized_vertices: object | None = None
    normalized_topology: object | None = None
    _boundaries_available: bool = field(
        default=False,
        repr=False,
        kw_only=True,
    )
    _periodic_shifts_available: bool = field(
        default=False,
        repr=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        """Validate the common contract and take ownership of aligned arrays."""

        if self.dimension not in (2, 3):
            raise ValueError('dimension must be 2 or 3')
        if self.mode not in ('standard', 'power'):
            raise ValueError('mode must be "standard" or "power"')
        if not isinstance(self.cells, list):
            raise ValueError('cells must be a list of dictionaries')
        if not all(isinstance(cell, dict) for cell in self.cells):
            raise ValueError('cells must be a list of dictionaries')

        sites_array = np.asarray(self.sites)
        if sites_array.ndim != 2:
            raise ValueError(
                f'sites must have shape (n, {self.dimension})'
            )
        n = int(sites_array.shape[0])
        sites = _readonly_float_array(
            sites_array,
            name='sites',
            shape=(n, self.dimension),
        )
        ids = _readonly_ids(self.ids, n=n)
        measures = _readonly_float_array(
            self.cell_measures,
            name='cell_measures',
            shape=(n,),
            nonnegative=True,
        )
        empty_mask = _readonly_mask(self.empty_mask, n=n)

        input_weights = None
        if self.input_weights is not None:
            input_weights = _readonly_float_array(
                self.input_weights,
                name='input_weights',
                shape=(n,),
            )

        backend_radii = None
        if self.backend_radii is not None:
            backend_radii = _readonly_float_array(
                self.backend_radii,
                name='backend_radii',
                shape=(n,),
                nonnegative=True,
            )

        shift = self.representation_shift
        if shift is not None:
            try:
                shift = float(shift)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    'representation_shift must be finite'
                ) from exc
            if not np.isfinite(shift):
                raise ValueError('representation_shift must be finite')

        if self.mode == 'standard':
            if input_weights is not None or backend_radii is not None:
                raise ValueError(
                    'standard mode cannot carry power representation arrays'
                )
            if shift is not None:
                raise ValueError(
                    'standard mode cannot carry a representation shift'
                )
        else:
            if backend_radii is None:
                raise ValueError('power mode requires backend_radii')
            if input_weights is None and shift is not None:
                raise ValueError(
                    'direct-radius power input cannot carry a '
                    'representation shift'
                )
            if input_weights is not None and shift is None:
                raise ValueError(
                    'weight-first power input requires a representation shift'
                )
            if input_weights is not None:
                assert shift is not None
                try:
                    expected_radii, _ = weights_to_radii(
                        input_weights,
                        weight_shift=shift,
                    )
                except ValueError as exc:
                    raise ValueError(
                        'weight-first representation metadata is inconsistent'
                    ) from exc
                if not np.array_equal(backend_radii, expected_radii):
                    raise ValueError(
                        'backend_radii do not match input_weights and '
                        'representation_shift'
                    )

        for name, value in (
            ('boundaries_available', self._boundaries_available),
            ('periodic_shifts_available', self._periodic_shifts_available),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f'{name} must be boolean')
        boundaries_available = bool(self._boundaries_available)
        periodic_shifts_available = bool(self._periodic_shifts_available)
        if periodic_shifts_available and not boundaries_available:
            raise ValueError(
                'periodic shifts cannot be available without boundaries'
            )

        raw_measures, raw_empty_mask = _aligned_raw_state(
            dimension=self.dimension,
            ids=ids,
            cells=self.cells,
            boundaries_available=boundaries_available,
            periodic_shifts_available=periodic_shifts_available,
        )
        if not np.array_equal(measures, raw_measures):
            raise ValueError(
                'cell_measures must match raw cells in input-ID order'
            )
        if not np.array_equal(empty_mask, raw_empty_mask):
            raise ValueError(
                'empty_mask must match raw cells in input-ID order'
            )

        object.__setattr__(self, 'sites', sites)
        object.__setattr__(self, 'ids', ids)
        object.__setattr__(self, 'cell_measures', measures)
        object.__setattr__(self, 'empty_mask', empty_mask)
        object.__setattr__(self, 'input_weights', input_weights)
        object.__setattr__(self, 'backend_radii', backend_radii)
        object.__setattr__(self, 'representation_shift', shift)
        object.__setattr__(
            self, '_boundaries_available', boundaries_available
        )
        object.__setattr__(
            self, '_periodic_shifts_available', periodic_shifts_available
        )

    @property
    def measure_kind(self) -> Literal['area', 'volume']:
        """Dimension-specific name of the values in ``cell_measures``."""

        return 'area' if self.dimension == 2 else 'volume'

    @property
    def boundary_kind(self) -> Literal['edges', 'faces']:
        """Dimension-specific raw-cell boundary key."""

        return 'edges' if self.dimension == 2 else 'faces'

    @property
    def has_tessellation_diagnostics(self) -> bool:
        """Whether dimension-specific tessellation diagnostics are present."""

        return self.tessellation_diagnostics is not None

    @property
    def has_normalized_vertices(self) -> bool:
        """Whether dimension-specific vertex normalization is present."""

        return self.normalized_vertices is not None

    @property
    def has_normalized_topology(self) -> bool:
        """Whether dimension-specific topology normalization is present."""

        return self.normalized_topology is not None

    @property
    def global_vertices(self) -> np.ndarray | None:
        """Global planar vertices from the available normalized output.

        This provisional convenience preserves the historical
        ``PlanarComputeResult`` access pattern. It is ``None`` when no planar
        normalization output is available.
        """

        if self.normalized_topology is not None:
            return self.normalized_topology.global_vertices
        if self.normalized_vertices is not None:
            return self.normalized_vertices.global_vertices
        return None

    @property
    def global_edges(self) -> list[dict[str, Any]] | None:
        """Global planar edges when topology normalization is available."""

        if self.normalized_topology is None:
            return None
        return self.normalized_topology.global_edges

    @property
    def has_boundaries(self) -> bool:
        """Whether raw boundary geometry was available to the builder."""

        return self._boundaries_available

    @property
    def has_periodic_shifts(self) -> bool:
        """Whether boundary records carry requested periodic image shifts."""

        return self._periodic_shifts_available

    def require_tessellation_diagnostics(self) -> object:
        """Return tessellation diagnostics or raise if none were computed."""

        if self.tessellation_diagnostics is None:
            raise ValueError(
                'tessellation diagnostics are not available; request '
                'diagnostics or enable a tessellation check'
            )
        return self.tessellation_diagnostics

    def require_normalized_vertices(self) -> object:
        """Return normalized vertices or raise if none were computed."""

        if self.normalized_vertices is None:
            raise ValueError(
                'normalized vertices are not available; request vertex or '
                'topology normalization'
            )
        return self.normalized_vertices

    def require_normalized_topology(self) -> object:
        """Return normalized topology or raise if none was computed."""

        if self.normalized_topology is None:
            raise ValueError(
                'normalized topology is not available; request topology '
                'normalization'
            )
        return self.normalized_topology

    def require_boundaries(self) -> list[list[dict[str, Any]]]:
        """Return boundary collections aligned with original input order.

        Hidden sites always have empty boundary collections, whether their raw
        records were omitted or contain an explicitly empty collection. When
        :attr:`has_periodic_shifts` is true, returned edge or face records
        contain ``adjacent_shift`` annotations.

        Raises:
            ValueError: If boundary geometry was not available, or if later
                mutation made the shared raw records internally inconsistent.
        """

        if not self._boundaries_available:
            raise ValueError(
                f'{self.boundary_kind} are not available; request boundary '
                'geometry during computation'
            )

        positions = {int(cell_id): pos for pos, cell_id in enumerate(self.ids)}
        aligned: list[list[dict[str, Any]]] = [
            [] for _ in range(int(self.ids.size))
        ]
        seen: set[int] = set()
        for cell in self.cells:
            cell_id = _raw_cell_id(cell)
            if cell_id not in positions:
                raise ValueError(
                    f'raw cell ID {cell_id} is not present in result ids'
                )
            if cell_id in seen:
                raise ValueError(f'duplicate raw cell ID {cell_id}')
            seen.add(cell_id)
            is_empty = _raw_cell_empty(cell, cell_id=cell_id)
            position = positions[cell_id]
            if is_empty != bool(self.empty_mask[position]):
                raise ValueError(
                    f'raw cell ID {cell_id} empty flag no longer matches '
                    'the result snapshot'
                )
            if self.boundary_kind not in cell:
                if is_empty:
                    continue
                raise ValueError(
                    f'raw cell ID {cell_id} has no {self.boundary_kind} data'
                )
            boundaries = cell[self.boundary_kind]
            _validate_boundary_records(
                boundaries,
                cell_id=cell_id,
                boundary_key=self.boundary_kind,
                is_empty=is_empty,
                periodic_shifts_available=self._periodic_shifts_available,
            )
            aligned[position] = boundaries

        missing_nonempty = [
            int(self.ids[position])
            for position in range(int(self.ids.size))
            if not bool(self.empty_mask[position])
            and int(self.ids[position]) not in seen
        ]
        if missing_nonempty:
            raise ValueError(
                'raw cells are missing for non-empty result IDs '
                f'{missing_nonempty}'
            )
        return aligned

    def _snapshot_state(self) -> dict[str, Any]:
        """Return exact state used for copying and serialization."""

        return {
            'dimension': self.dimension,
            'domain': self.domain,
            'mode': self.mode,
            'sites': self.sites,
            'ids': self.ids,
            'cells': self.cells,
            'cell_measures': self.cell_measures,
            'empty_mask': self.empty_mask,
            'input_weights': self.input_weights,
            'backend_radii': self.backend_radii,
            'representation_shift': self.representation_shift,
            'tessellation_diagnostics': self.tessellation_diagnostics,
            'normalized_vertices': self.normalized_vertices,
            'normalized_topology': self.normalized_topology,
            '_boundaries_available': self._boundaries_available,
            '_periodic_shifts_available': self._periodic_shifts_available,
        }

    def __deepcopy__(self, memo: dict[int, Any]) -> TessellationResult:
        """Deep-copy snapshots and mutable contents without revalidation."""

        import copy

        result = object.__new__(type(self))
        memo[id(self)] = result
        state = copy.deepcopy(self._snapshot_state(), memo)
        _restore_tessellation_result_state(result, state)
        return result

    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[object, ...]]:
        """Use a version-independent pickle restoration path."""

        return (
            _restore_tessellation_result,
            (type(self), self._snapshot_state()),
        )


def _restore_tessellation_result(
    cls: type[TessellationResult],
    state: dict[str, Any],
) -> TessellationResult:
    """Rebuild a serialized result without rerunning constructor checks."""

    result = object.__new__(cls)
    _restore_tessellation_result_state(result, state)
    return result


def _restore_tessellation_result_state(
    result: TessellationResult,
    state: dict[str, Any],
) -> None:
    """Restore snapshot state and re-establish owned read-only arrays."""

    array_fields = {
        'sites',
        'ids',
        'cell_measures',
        'empty_mask',
        'input_weights',
        'backend_radii',
    }
    for name, value in state.items():
        if name in array_fields and value is not None:
            value = np.array(value, copy=True, order='C')
            value.setflags(write=False)
        object.__setattr__(result, name, value)


def _raw_cell_id(cell: dict[str, Any]) -> int:
    """Return a raw cell ID without accepting truncating conversions."""

    if 'id' not in cell:
        raise ValueError('raw cell record is missing its id')
    value = cell['id']
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('raw cell IDs must be integers')
    try:
        return int(index(value))
    except TypeError as exc:
        raise ValueError('raw cell IDs must be integers') from exc


def _raw_cell_empty(cell: dict[str, Any], *, cell_id: int) -> bool:
    """Return a raw empty flag without accepting truthy non-booleans."""

    value = cell.get('empty', False)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f'raw cell ID {cell_id} empty flag must be boolean'
        )
    return bool(value)


def _raw_cell_measure(
    cell: dict[str, Any],
    *,
    cell_id: int,
    measure_key: str,
) -> float:
    """Return one finite, non-negative raw cell measure."""

    if measure_key not in cell:
        raise ValueError(
            f'raw cell ID {cell_id} is missing {measure_key}'
        )
    try:
        measure = float(cell[measure_key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'raw cell ID {cell_id} has invalid {measure_key}'
        ) from exc
    if not np.isfinite(measure) or measure < 0.0:
        raise ValueError(
            f'raw cell ID {cell_id} has invalid {measure_key}'
        )
    return measure


def _validate_boundary_records(
    boundaries: object,
    *,
    cell_id: int,
    boundary_key: str,
    is_empty: bool,
    periodic_shifts_available: bool,
) -> None:
    """Validate one raw boundary collection without copying it."""

    if not isinstance(boundaries, list):
        raise ValueError(
            f'raw cell ID {cell_id} {boundary_key} must be a list'
        )
    if is_empty and boundaries:
        raise ValueError(
            f'raw empty cell ID {cell_id} {boundary_key} must be empty'
        )
    for boundary in boundaries:
        if not isinstance(boundary, dict):
            raise ValueError(
                f'raw cell ID {cell_id} {boundary_key} records must be '
                'dictionaries'
            )
        if periodic_shifts_available and 'adjacent_shift' not in boundary:
            raise ValueError(
                f'raw cell ID {cell_id} has a {boundary_key} record '
                'without adjacent_shift'
            )
        if not periodic_shifts_available and 'adjacent_shift' in boundary:
            raise ValueError(
                f'raw cell ID {cell_id} has adjacent_shift but periodic '
                'shifts are unavailable'
            )


def _aligned_raw_state(
    *,
    dimension: Literal[2, 3],
    ids: np.ndarray,
    cells: list[dict[str, Any]],
    boundaries_available: bool,
    periodic_shifts_available: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate raw records and derive ID-aligned measures and empty state."""

    positions = {
        int(cell_id): position for position, cell_id in enumerate(ids)
    }
    measures = np.zeros(ids.size, dtype=np.float64)
    empty_mask = np.ones(ids.size, dtype=np.bool_)
    seen: set[int] = set()
    measure_key = 'area' if dimension == 2 else 'volume'
    boundary_key = 'edges' if dimension == 2 else 'faces'

    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError('cells must be a list of dictionaries')
        cell_id = _raw_cell_id(cell)
        if cell_id not in positions:
            raise ValueError(
                f'raw cell ID {cell_id} is not present in input ids'
            )
        if cell_id in seen:
            raise ValueError(f'duplicate raw cell ID {cell_id}')
        seen.add(cell_id)

        measure = _raw_cell_measure(
            cell,
            cell_id=cell_id,
            measure_key=measure_key,
        )
        is_empty = _raw_cell_empty(cell, cell_id=cell_id)
        if is_empty and measure != 0.0:
            raise ValueError(
                f'raw empty cell ID {cell_id} must have zero {measure_key}'
            )
        if boundaries_available:
            if not is_empty and boundary_key not in cell:
                raise ValueError(
                    f'raw cell ID {cell_id} is missing requested '
                    f'{boundary_key}'
                )
            _validate_boundary_records(
                cell.get(boundary_key, []),
                cell_id=cell_id,
                boundary_key=boundary_key,
                is_empty=is_empty,
                periodic_shifts_available=periodic_shifts_available,
            )
        elif boundary_key in cell:
            raise ValueError(
                f'raw cell ID {cell_id} contains {boundary_key} but '
                'boundaries are unavailable'
            )

        position = positions[cell_id]
        empty_mask[position] = is_empty
        measures[position] = measure

    return measures, empty_mask


def _build_tessellation_result(
    *,
    dimension: Literal[2, 3],
    domain: object,
    mode: Literal['standard', 'power'],
    sites: np.ndarray,
    ids: Sequence[int] | np.ndarray | None,
    cells: list[dict[str, Any]],
    power_input: ResolvedPowerInput,
    tessellation_diagnostics: object | None = None,
    normalized_vertices: object | None = None,
    normalized_topology: object | None = None,
    boundaries_available: bool,
    periodic_shifts_available: bool,
) -> TessellationResult:
    """Build a common result from final raw cells without further computation.

    Raw cells are matched to original input positions by their final external
    IDs. Missing raw cells become zero-measure entries in ``empty_mask``.
    """

    if dimension not in (2, 3):
        raise ValueError('dimension must be 2 or 3')
    if mode not in ('standard', 'power'):
        raise ValueError('mode must be "standard" or "power"')
    if not isinstance(power_input, ResolvedPowerInput):
        raise ValueError('power_input must be a ResolvedPowerInput')
    if not isinstance(cells, list):
        raise ValueError('cells must be a list of dictionaries')
    for name, value in (
        ('boundaries_available', boundaries_available),
        ('periodic_shifts_available', periodic_shifts_available),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be boolean')
    boundaries_available = bool(boundaries_available)
    periodic_shifts_available = bool(periodic_shifts_available)
    if periodic_shifts_available and not boundaries_available:
        raise ValueError(
            'periodic shifts cannot be available without boundaries'
        )

    sites_array = np.asarray(sites)
    if sites_array.ndim != 2 or sites_array.shape[1] != dimension:
        raise ValueError(f'sites must have shape (n, {dimension})')
    n = int(sites_array.shape[0])
    if ids is None:
        ids_array = np.arange(n, dtype=np.int64)
    else:
        ids_array = _readonly_ids(np.asarray(ids), n=n)

    measures, empty_mask = _aligned_raw_state(
        dimension=dimension,
        ids=ids_array,
        cells=cells,
        boundaries_available=boundaries_available,
        periodic_shifts_available=periodic_shifts_available,
    )

    return TessellationResult(
        dimension=dimension,
        domain=domain,
        mode=mode,
        sites=sites_array,
        ids=ids_array,
        cells=cells,
        cell_measures=measures,
        empty_mask=empty_mask,
        input_weights=power_input.input_weights,
        backend_radii=power_input.backend_radii,
        representation_shift=power_input.representation_shift,
        tessellation_diagnostics=tessellation_diagnostics,
        normalized_vertices=normalized_vertices,
        normalized_topology=normalized_topology,
        _boundaries_available=boundaries_available,
        _periodic_shifts_available=periodic_shifts_available,
    )


__all__ = ['TessellationResult']
