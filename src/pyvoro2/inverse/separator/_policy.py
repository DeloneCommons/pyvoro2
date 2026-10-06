"""Owned positional model binding, ordered projection, and inspection.

Policy is deliberately separate from ADR 0014 observation/source identity.
Numerical consumers use the typed model; the mapping is an inspection adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import MappingProxyType

import numpy as np

from ._identity import (
    _ObservationBoundResult, _require_observation_association, _row_ids,
)
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


def _readonly_policy_array(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.flags.writeable:
        array = array.copy()
        array.setflags(write=False)
    return array


def _row_model(model, index):
    def row(term):
        if term is None:
            return None
        return replace(term, **{
            item.name: getattr(term, item.name)[index].item()
            for item in fields(term)
            if isinstance(getattr(term, item.name), np.ndarray)
        })
    return replace(model, feasible=row(model.feasible),
                   penalties=tuple(row(term) for term in model.penalties))


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
                'hard_constraint': _term_policy(
                    model.feasible, self.observations.measurement,
                ),
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
                raise ValueError(
                    f'{type(term).__name__}.{item.name} must have shape ({count},)')
    reference = model.regularization.reference
    if reference is not None and reference.shape != (observations.n_points,):
        raise ValueError('L2Regularization.reference must have shape (n_points,)')
    # Reconstruct the complete typed template so even deliberate mutation of
    # a caller's array flags cannot change a bound numerical policy.
    return _BoundPolicy(observations, replace(
        model, mismatch=replace(model.mismatch),
        feasible=None if model.feasible is None else replace(model.feasible),
        penalties=tuple(replace(term) for term in model.penalties),
        regularization=replace(model.regularization),
    ))


class _PolicyAccess:
    __slots__ = ()

    def _require_policy(self):
        policy = getattr(self, '_bound_policy', None)
        if policy is None:
            raise ValueError('result has no authoritative resolved model policy')
        return policy

    @property
    def resolved_policy(self):
        return self._require_policy().view

    @property
    def mismatch_space(self):
        return self._require_policy().mismatch_space

    @property
    def hard_constraint_space(self):
        return self._require_policy().hard_constraint_space

    @property
    def penalty_spaces(self):
        return self._require_policy().penalty_spaces

    @property
    def mismatch_target(self):
        observations = self._require_policy().observations
        return _readonly_policy_array(
            observations.target_fraction if self.mismatch_space == 'fraction'
            else observations.target_position,
        )


class _PolicyStorage(_PolicyAccess):
    __slots__ = ('_bound_policy',)


class _PolicyResultStorage(_ObservationBoundResult, _PolicyAccess):
    __slots__ = ('_bound_policy',)

    @property
    def mismatch_predicted(self):
        self._require_policy()
        return _readonly_policy_array(
            self.predicted_fraction if self.mismatch_space == 'fraction'
            else self.predicted_position,
        )

    @property
    def mismatch_residuals(self):
        policy = self._require_policy()
        if self.weights is None:
            return None
        from .problem import _residual_diagnostic

        return _residual_diagnostic(
            policy.observations, self.weights, space=self.mismatch_space,
        ).value


class _PolicyBindingInit:
    def __get__(self, instance, owner=None):
        return None if instance is None else getattr(instance, '_bound_policy', None)


def _bind_result_policy(result, policy):
    origin = getattr(result, '_originating_observations', None)
    if origin is not None:
        _require_observation_association(
            origin, policy.observations, context='resolved model policy',
        )
    existing = getattr(result, '_bound_policy', None)
    if existing is not None and existing.view != policy.view:
        raise ValueError('result resolved policy does not match its originating model')
    object.__setattr__(result, '_bound_policy', policy)
    return result


def _policy_getstate(value):
    return [*(getattr(value, item.name) for item in fields(value)),
            getattr(value, '_bound_policy', None)]


def _policy_setstate(value, state):
    value_fields = fields(value)
    if len(state) not in (len(value_fields), len(value_fields) + 1):
        raise ValueError('invalid resolved policy reconstruction state')
    for item, field_value in zip(value_fields, state):
        object.__setattr__(value, item.name, field_value)
    if len(state) > len(value_fields) and state[-1] is not None:
        object.__setattr__(value, '_bound_policy', state[-1])
