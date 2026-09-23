"""Refusal policy is separate from completed geometric classification."""

from fractions import Fraction

import pytest

from pyvoro2._internal.spatial.wp5_common import (
    WP5Budget,
    WP5Failure,
    WP5Limits,
)


def test_complete_candidate_region_refused_before_work():
    budget = WP5Budget(WP5Limits(candidate_limit=4))
    with pytest.raises(WP5Failure) as raised:
        budget.candidates(5, stage='producer')
    assert raised.value.code == 'WP5_RESOURCE_LIMIT'
    assert raised.value.context['required'] == 5
    assert budget.work == 0


def test_work_guard_does_not_return_a_successful_prefix():
    budget = WP5Budget(WP5Limits(work_limit=2))
    budget.charge(2, stage='polytope')
    with pytest.raises(WP5Failure, match='work') as raised:
        budget.charge(stage='polytope')
    assert raised.value.code == 'WP5_RESOURCE_LIMIT'


def test_exact_numerator_and_denominator_bit_guards():
    budget = WP5Budget(WP5Limits(bit_limit=8))
    budget.fraction(Fraction(127, 128), stage='ideal')
    for value in (Fraction(256), Fraction(1, 256)):
        with pytest.raises(WP5Failure) as raised:
            budget.fraction(value, stage='ideal')
        assert raised.value.code == 'WP5_RESOURCE_LIMIT'


def test_failure_retains_source_context_without_reclassification():
    error = WP5Failure('WP5_IMAGE_INCONSISTENT', 'no histories',
                       source_id=3, token=17)
    assert error.context == {'source_id': 3, 'token': 17}
    assert str(error) == 'no histories'
