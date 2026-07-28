# 0008 — Separator solver and linear-backend selection

- **Status:** Accepted
- **Date:** 2026-07-30
- **Related issue:** [#36 — Correct and freeze the separator inverse objective contract](https://github.com/DeloneCommons/pyvoro2/issues/36)
- **Related decisions:** [ADR 0007](0007-separator-objective-contract.md)
- **Related plan:** [v0.8 remediation execution plan](../plans/v0.8-remediation.md)

## Context

The prerelease separator API used one overloaded ``solver`` string. Values such
as ``analytic`` and ``admm`` named optimization methods, ``sparse`` named a
linear-algebra backend, and ``auto`` selected both. ADMM also switched from
dense NumPy to SciPy sparse linear algebra above a private component-size
threshold. Consequently, an otherwise identical call could acquire a new
optional dependency and exception behavior solely because a component crossed
that threshold. Explicit ADMM on a quadratic model could also be replaced by a
direct solve, so the requested method was not always the method executed.

This ambiguity obstructs reproducibility, independent backend testing, and the
universal success contract in ADR 0007. The API is provisional and unreleased,
so preserving defective compatibility would add complexity without protecting a
working contract.

## Decision

Optimization method and linear-algebra backend are independent keyword-only
parameters on the high-level static fit:

```python
solver: Literal["direct", "admm"] = "direct"
linear_backend: Literal["dense", "sparse"] = "dense"
admm_max_iter: int = 2000
admm_rho: float = 1.0
admm_abs_tol: float = 1e-6
admm_rel_tol: float = 1e-5
```

The active-set wrapper exposes the same choices with ``fit_`` prefixes for its
inner fixed-row fit. Public solver-configuration classes are not introduced.
The flat keyword interface is the authoritative reproducible surface.

The four supported combinations have exact meanings:

- ``direct + dense``: certified direct quadratic solve using NumPy;
- ``direct + sparse``: certified direct quadratic solve using SciPy;
- ``admm + dense``: actual ADMM using dense NumPy weight-system solves;
- ``admm + sparse``: actual ADMM using sparse SciPy weight-system solves.

The default is ``direct + dense`` because the ordinary small-system model is
quadratic, the direct method is more accurate and efficient there, and NumPy is
a core dependency. ``direct`` rejects Huber mismatch, active scalar penalties,
or hard restrictions with an actionable error directing the caller to ADMM.
An explicit ADMM request executes ADMM even for a purely quadratic model. A
direct quadratic solve may initialize ADMM, but may not replace the requested
method. A quadratic ADMM result is successful only after the final returned
weights pass ADR 0007's universal quadratic certificate.

The requirement that explicit ADMM execute applies when the resolved problem
has a non-singleton component to solve. Empty observation sets and models whose
coupling components are all singletons are no-work fits: their weights follow
the documented reference or gauge convention without invoking a component
solver. They report ``solver='none'``, ``linear_backend=None``, and
``n_iter=0`` rather than attributing the closed-form result to the requested
method or matrix backend.

``linear_backend='dense'`` never imports SciPy and never switches to sparse by
problem size or package availability. ``linear_backend='sparse'`` explicitly
requires SciPy and fails early with an actionable ``ImportError`` when it is
missing. Production candidate generation does not silently cross between dense
and sparse backends. Normal-to-augmented recovery remains within the selected
backend. Bounded exact certification helpers are not production backends and
remain permitted under ADR 0007.

Remove the prerelease values ``auto``, ``analytic``, and ``sparse`` from the
``solver`` parameter, and replace the unqualified ADMM parameters ``max_iter``,
``rho``, ``tol_abs``, and ``tol_rel`` with the prefixed names above. Do not add
aliases, deprecation shims, compatibility wrappers, or a legacy mode. A future
model-default convenience may be considered after R2, but no automatic mode is
part of v0.8 R1.

Public results, termination views, reports, and JSON-friendly records expose
method and backend separately:

- ``solver`` is ``direct``, ``admm``, ``external``, or ``none`` as appropriate;
- ``linear_backend`` is ``dense``, ``sparse``, or ``None`` when no internal
  matrix backend ran.

``n_iter`` records completed ADMM iterations even when the iterate later fails
the universal quadratic certificate or another supported post-iteration
numerical check. A structured ``numerical_failure`` must not rewrite completed
iterative work to zero. Direct component failures that occur before a certified
candidate is returned may still report zero iterations.

The provisional ADMM defaults above are retained during R1. They are not tuned
until R2 replaces and certifies the scalar proximal solver. Users control the
algorithm, backend, ADMM penalty, iteration budget, and ADMM stopping
tolerances. They do not control the numerical criteria that define
``status='optimal'``.

## Consequences

- Ordinary small quadratic fits remain simple and dependency-free.
- Solver choice and optional dependency behavior are explicit and reproducible.
- Dense calls behave identically whether SciPy is installed or not.
- Explicit ADMM becomes independently testable rather than an alias for a
  direct solve.
- Sparse direct and sparse ADMM become deliberate opt-in capabilities.
- Current call sites and documentation require a prerelease migration, but no
  compatibility layer is carried into later remediation work.
- Large dense requests may fail honestly rather than silently changing backend;
  resource policy beyond the existing safe implementation remains a separate
  decision if evidence requires it.

## Alternatives considered

### Retain ``auto`` but forbid dependency switching

Rejected for R1 because the parameter would still mix model capability and
method selection while the ADMM path is awaiting R2 certification. A narrowly
defined future ``model_default`` convenience may be reconsidered later.

### Expose public solver configuration objects

Rejected for the high-level API. Flat keyword-only parameters are clearer in
signatures, IDE completion, examples, and configuration files, and avoid
freezing premature public classes before 1.0. A private internal configuration
object remains an implementation option.

### Opportunistically use SciPy when installed

Rejected because identical calls would execute different algorithms in
different environments and optional-package presence would affect scientific
reproducibility.

### Preserve the old values through aliases or deprecation warnings

Rejected because v0.8 is unreleased, the old contract is internally ambiguous,
and compatibility support would perpetuate behavior that does not work
correctly.
