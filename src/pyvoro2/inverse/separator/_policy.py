"""Owned positional model binding, ordered projection, and inspection.

Policy is deliberately separate from ADR 0014 observation/source identity.
Numerical consumers use the typed model; the mapping is an inspection adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import MappingProxyType

import numpy as np

from ._identity import _row_ids
from .model import FitModel


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _row_parameter(value):
    if isinstance(value, np.ndarray):
        return {'kind': 'rows', 'values': value.tolist()}
    return {'kind': 'uniform', 'value': value}


def _term_policy(term, source_space):
    if term is None:
        return None
    result = {
        'family': type(term).__name__, 'space': term.space or source_space,
        'parameters': {
            item.name: _row_parameter(getattr(term, item.name))
            for item in fields(term) if item.name not in ('space', 'applicable')
        },
    }
    if hasattr(term, 'applicable'):
        result['applicable'] = _row_parameter(term.applicable)
    return result


def _project_term(term, indices):
    if term is None:
        return None
    return replace(term, **{
        item.name: getattr(term, item.name)[indices]
        for item in fields(term)
        if isinstance(getattr(term, item.name), np.ndarray)
    })


def _expand(value, count, *, dtype=float):
    return (np.full(count, value, dtype=dtype) if np.ndim(value) == 0
            else np.asarray(value, dtype=dtype))


@dataclass(frozen=True, slots=True)
class _BoundPolicy:
    observations: object
    model: FitModel

    def project(self, selection):
        """Use one local ordered selection for observations and every row field."""
        array = np.asarray(selection)
        if array.size == 0:
            array = np.empty(0, dtype=np.int64)
        indices = np.arange(self.observations.n_constraints)[array]
        if array.dtype.kind == 'b':
            observations = self.observations.subset(array)
        else:
            observations = replace(self.observations, **{
                name: getattr(self.observations, name)[indices]
                for name in ('i', 'j', 'shifts', 'target', 'confidence',
                             'distance', 'distance2', 'delta', 'target_fraction',
                             'target_position', 'input_index', 'explicit_shift')
            })
        return _bind_policy(observations, replace(
            self.model,
            feasible=_project_term(self.model.feasible, indices),
            penalties=tuple(_project_term(term, indices)
                            for term in self.model.penalties),
        ))

    @property
    def mismatch_space(self):
        return self.model.mismatch.space or self.observations.measurement

    @property
    def hard_constraint_space(self):
        term = self.model.feasible
        return None if term is None else term.space or self.observations.measurement

    @property
    def penalty_spaces(self):
        return tuple(term.space or self.observations.measurement
                     for term in self.model.penalties)

    @property
    def applicable(self):
        term = self.model.feasible
        count = self.observations.n_constraints
        return (_expand(False, count, dtype=bool) if term is None
                else _expand(term.applicable, count, dtype=bool))

    @property
    def view(self):
        model = self.model
        regularization = model.regularization
        reference = regularization.reference
        return _freeze({
            'row_ids': _row_ids(self.observations),
            'model_spaces': {
                'mismatch': self.mismatch_space,
                'hard_constraint': self.hard_constraint_space,
                'penalties': self.penalty_spaces,
            },
            'model_policy': {
                'mismatch': _term_policy(model.mismatch, self.observations.measurement),
                'hard_constraint': _term_policy(model.feasible, self.observations.measurement),
                'penalties': [_term_policy(term, self.observations.measurement)
                              for term in model.penalties],
                'regularization': {
                    'family': 'L2Regularization',
                    'strength': regularization.strength,
                    'reference': ({'kind': 'implicit_zero'} if reference is None else
                                  {'kind': 'sites', 'values': reference.tolist()}),
                },
            },
        })


def _bind_policy(observations, model):
    """Check exact candidate lengths before any subset or numerical work."""
    if not isinstance(model, FitModel):
        raise ValueError('model must be a FitModel')
    count = observations.n_constraints
    for term in (model.mismatch, model.feasible, *model.penalties):
        if term is None:
            continue
        for item in fields(term):
            value = getattr(term, item.name)
            if isinstance(value, np.ndarray) and value.shape != (count,):
                raise ValueError(f'{type(term).__name__}.{item.name} must have shape ({count},)')
    reference = model.regularization.reference
    if reference is not None and reference.shape != (observations.n_points,):
        raise ValueError('L2Regularization.reference must have shape (n_points,)')
    return _BoundPolicy(observations, model)
