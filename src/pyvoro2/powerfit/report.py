"""Compatibility exports for separator reports."""

from __future__ import annotations

from ..inverse.separator.report import (
    build_active_set_report,
    build_fit_report,
    build_realized_report,
    dumps_report_json,
    write_report_json,
)

__all__ = [
    'build_fit_report',
    'build_realized_report',
    'build_active_set_report',
    'dumps_report_json',
    'write_report_json',
]
