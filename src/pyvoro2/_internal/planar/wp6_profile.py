"""Reviewed source/schema boundary of the compact ordinary planar witness.

The literal source digest is deliberately separate from CMake's measured
closure.  Changing it requires qualification, not automatic regeneration.
Runtime floating-point state is checked on every native entry.
"""
from __future__ import annotations

from collections.abc import Mapping

SCHEMA = "pyvoro2.planar.occurrences.v1"
SOURCE_SHA256 = "106c866bd221e2ed5c4e7942c04d756c0d4519ccd0692f90ffdba18bd93e4acf"
SUPPORTED_COHORTS = frozenset({"linux-x86_64-gcc13.3-sse2"})


def require_supported_profile(profile: Mapping) -> None:
    """Refuse a mismatched native producer before consuming any provenance."""
    def refuse(reason: str, detail: str) -> None:
        raise RuntimeError(f"planar_certification:profile:{reason}: {detail}")

    if not isinstance(profile, Mapping):
        refuse("schema", "native witness profile is not a mapping")
    if profile.get("schema") != SCHEMA:
        refuse("schema", "Python/native occurrence schema differs")
    if (profile.get("source_sha256") != SOURCE_SHA256
            or profile.get("source_supported") is not True):
        refuse("source", "native source closure differs from reviewed Python contract")
    if (profile.get("cohort") not in SUPPORTED_COHORTS
            or profile.get("cohort_supported") is not True):
        refuse("cohort", "compiler/target cohort is not qualified")
    expected = {"flt_eval_method": 0, "int_bits": 32, "uint_bits": 32,
                "double_digits": 53, "round_to_nearest": True,
                "gradual_underflow": True, "fp_contract": "off",
                "fast_math": False, "lto": False, "qualified": True}
    if any(profile.get(key) != value for key, value in expected.items()):
        refuse("evaluation", "native arithmetic/build predicate is not satisfied")
    build = profile.get("build_sha256")
    if not isinstance(build, str) or len(build) != 64:
        refuse("build", "native build identity is absent")
    try:
        int(build, 16)
    except ValueError:
        refuse("build", "native build identity is malformed")


validate_profile = require_supported_profile
