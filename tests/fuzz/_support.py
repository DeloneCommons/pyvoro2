from __future__ import annotations

import numpy as np


def rng_for_run(seed: int, run: int) -> np.random.Generator:
    """Return the deterministic random-number generator for one fuzz run."""
    mixed = (seed + 0x9E3779B97F4A7C15 + 104729 * int(run)) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(mixed)
