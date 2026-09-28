"""Guard overridable domain operations before numerical continuation."""

from __future__ import annotations

import operator

from .native_runtime import checked_call


def call_domain_method(domain, name, /, *args, **kwargs):
    """Check method acquisition separately from the method's return."""

    method = checked_call(getattr, domain, name)
    return checked_call(method, *args, **kwargs)


def periodic_axis(domain, axis: int) -> bool:
    """Read one flag without combining foreign lookup and truth conversion."""

    periodic = checked_call(getattr, domain, 'periodic')
    value = checked_call(operator.getitem, periodic, axis)
    return checked_call(bool, value)
