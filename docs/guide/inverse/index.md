# Inverse fitting

Given fixed sites and observed pairwise separator positions, fit mathematical
power weights with the stable high-level `pyvoro2.inverse` API. The smallest
example uses two sites two length units apart. A separator one quarter of the
way from site 0 to site 1 lies at `x = 0.5`.

For the power convention $\lVert x-p_i\rVert^2-w_i$, equal powers at that
separator imply

$$
w_0-w_1 = 2d^2(t-1/2) = 2(2^2)(1/4-1/2) = -2.
$$

```python
import numpy as np
from pyvoro2.inverse import (
    resolve_separator_observations,
    fit_weights_from_separators,
)

points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
observations = resolve_separator_observations(
    points, [(0, 1, 0.25)], measurement='fraction',
)
fit = fit_weights_from_separators(points, observations)
assert fit.status == 'optimal'
np.testing.assert_allclose(fit.weights[0] - fit.weights[1], -2.0, atol=1e-12)
np.testing.assert_allclose(fit.predicted_fraction, [0.25], atol=1e-12)
np.testing.assert_allclose(fit.predicted_position, [0.5], atol=1e-12)
print(fit.status, fit.weights[0] - fit.weights[1])  # optimal -2.0
```

The resolver returns `SeparatorObservations`: validated endpoints, connector
geometry and targets. With no periodic domain, the connector joins the given
Cartesian sites directly. The default fixed-observation fit uses squared
measurement mismatch and a direct dense solve. Check the returned status
before using weights; inconsistent observations need not admit zero residual.

Only the **weight difference** is identified here. Adding one constant to all
weights leaves the diagram unchanged; absolute fitted weights are a gauge
choice. Disconnected observation components have additional unidentified
relative offsets, which can change the complete diagram.

## From a fit to geometry

**Algebraic fit does not establish a realized contact.** A third site can hide
a fitted separator, and a periodic pair may meet through a different image.
Compute the power diagram with `weights=fit.weights` in a supported domain,
and check requested-image realization separately when contacts matter.
The [separator-fitting guide](separator-fitting.md#canonical-downstream-integration)
shows external IDs, explicit periodic shifts, forward computation and matching.

The same stable fitting entry points accept 2D or 3D points; domain support
remains dimension-specific. This workflow fits separator observations, not
prescribed areas or volumes. Advanced model, realization and report helpers in
`pyvoro2.inverse.separator` are provisional; the active-set outer loop is
experimental and has no universal convergence guarantee.

Continue with the [full guide](separator-fitting.md),
[high-level API](../../reference/inverse/index.md),
[advanced API](../../reference/inverse/separator.md),
[Theory](../../theory/separator-inverse.md), or
[inverse notebooks](../../examples/index.md#inverse-fitting).

## Research article

**Powerfit research article — current author manuscript (submitted for peer review)**:
[current HTML](https://delonecommons.github.io/powerfit/article.html) and
[current PDF](https://delonecommons.github.io/powerfit/manuscript.pdf).
See the [reproducibility materials](https://github.com/DeloneCommons/powerfit/blob/main/docs/reproducibility.md)
and [version/citation guidance](https://github.com/DeloneCommons/powerfit/blob/main/docs/archives.md).
The current manuscript, the frozen powerfit v2.0.0 archive, and its reported
pyvoro2 0.6.3 software baseline are distinct identities. The example above uses
the current pyvoro2 API; the article provides research context and full proofs.
