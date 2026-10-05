"""Preferred realization-aware separator workflow over the existing engine."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from ...domains import Box as Box3D, OrthorhombicCell, PeriodicCell
from ...planar.domains import Box as Box2D, RectangularCell
from .active import (
    ActiveSetOptions,
    SelfConsistentPowerFitResult,
    solve_self_consistent_power_weights,
)
from .constraints import SeparatorObservations
from .model import FitModel


def fit_self_consistent_weights_from_separators(
    points: np.ndarray,
    constraints: SeparatorObservations | list[tuple] | tuple[tuple, ...],
    *,
    measurement: Literal['fraction', 'position'] = 'fraction',
    domain: Box2D | RectangularCell | Box3D | OrthorhombicCell | PeriodicCell,
    ids: Sequence[int | np.integer] | np.ndarray | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    confidence: list[float] | tuple[float, ...] | np.ndarray | None = None,
    model: FitModel | None = None,
    r_min: float = 0.0,
    weight_shift: float | None = None,
    fit_solver: Literal['direct', 'admm'] = 'direct',
    fit_linear_backend: Literal['dense', 'sparse'] = 'dense',
    fit_admm_max_iter: int = 2000,
    fit_admm_rho: float = 1.0,
    fit_admm_abs_tol: float = 1e-6,
    fit_admm_rel_tol: float = 1e-5,
    max_outer_iter: int = 25,
    return_cells: bool = False,
    return_boundary_measure: bool = False,
    return_tessellation_diagnostics: bool = False,
    tessellation_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'diagnose',
    connectivity_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'warn',
    unaccounted_pair_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'warn',
) -> SelfConsistentPowerFitResult:
    """Fit weights while refining which supplied separators are realized.

    This Provisional workflow starts with every candidate active and uses the
    existing empirical active engine with its default hysteresis and cycle
    policy. ``max_outer_iter`` is a positive exact non-Boolean index scalar.
    Models requiring ADMM need explicit ``fit_solver='admm'``; SciPy is loaded
    only for an explicitly selected sparse backend.

    Inspect ``outer_termination``, ``inner_fit.status`` and
    ``final_state_available`` separately. Cycles and iteration limits can have
    an available final state. Without usable final weights, realization,
    candidate diagnostics and records are ``None`` even when outputs are
    requested. Output switches control optional final layers, not internal
    tessellation or mandatory semantic certification.

    Raw and resolved observations retain the engine's exact source binding and
    validation. Input, policy, diagnostic-raise and native certificate failures
    propagate as exceptions. Candidate generation and global convergence are
    not provided. The fixed ``fit_weights_from_separators`` is a separate solve.

    The identical shared result retains Experimental path/counter members and
    report-v2 data. ``history`` is ``None`` on this route. Final-state inspection
    is Provisional regardless of which namespace imported the result class.
    """
    return solve_self_consistent_power_weights(
        points,
        constraints,
        measurement=measurement,
        domain=domain,
        ids=ids,
        index_mode=index_mode,
        image=image,
        image_search=image_search,
        confidence=confidence,
        model=model,
        active0=None,
        options=ActiveSetOptions(max_iter=max_outer_iter),
        r_min=r_min,
        weight_shift=weight_shift,
        fit_solver=fit_solver,
        fit_linear_backend=fit_linear_backend,
        fit_admm_max_iter=fit_admm_max_iter,
        fit_admm_rho=fit_admm_rho,
        fit_admm_abs_tol=fit_admm_abs_tol,
        fit_admm_rel_tol=fit_admm_rel_tol,
        return_history=False,
        return_cells=return_cells,
        return_boundary_measure=return_boundary_measure,
        return_tessellation_diagnostics=return_tessellation_diagnostics,
        tessellation_check=tessellation_check,
        connectivity_check=connectivity_check,
        unaccounted_pair_check=unaccounted_pair_check,
    )
