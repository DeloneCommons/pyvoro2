"""Arithmetic-free structured locate failure used at the first public guard."""

from .ghost_failure import _bounded


_CODES = frozenset({
    'LOCATE_BACKEND_INSERTION', 'LOCATE_NATIVE_UNSUPPORTED',
    'LOCATE_PROVENANCE_AMBIGUOUS', 'LOCATE_PROVENANCE_INCONSISTENT',
    'LOCATE_CERTIFICATION_RESOURCE', 'LOCATE_METADATA_UNREPRESENTABLE',
})


class LocateFailure(ValueError):
    """Private class implementing the provisional inspectable failure protocol."""

    def __init__(self, code, message, *, stage, query_index=None, **details):
        if code not in _CODES:
            raise ValueError('unknown locate failure code')
        self.code, self.stage, self.query_index = code, stage, query_index
        self.details = _bounded(details)
        super().__init__(f'{code} [{stage}]: {str(message)[:512]}')
