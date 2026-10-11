# Python/native boundary: validation, safety and unverified assumptions

**Status:** v0.8-based maintainer review. Separate **implemented safeguards**, **their limits**, and **questions for investigation**. This note supplements, but does not replace, [ADR 0010](../../../docs/development/decisions/0010-native-construction-preconditions.md), [ADR 0011](../../../docs/development/decisions/0011-strict-input-and-ownership-contract.md), and [ADR 0013](../../../docs/development/decisions/0013-central-generator-preparation-and-backend-safety.md).

## Responsibility split

| Python (public-source semantics and preparation) | Native/C++ (converted-value backstop and Voro++ construction safety) |
| --- | --- |
| Validate original input categories: exact integer vs Boolean, finite-real arrays, scalar string choices, flag types; preserve caller-owned data through canonical immutable copies | Check already converted arrays' shapes/counts, finiteness, internal `int` ID uniqueness/ranges, nonnegative finite radii |
| Enforce `standard`/`power` input relationships; validate mathematical weights vs radii and calculate backend representation | Validate the resulting backend radii; it does not receive original weight-input semantics |
| Validate public external IDs (unique, nonnegative, signed int64), assign internal `0..n-1` IDs | Recheck internal IDs and the planar synthetic ghost-ID reservation |
| Validate user domain and periodic basis; transform/remap coordinates; choose blocks | Validate native box bounds or six internal periodic parameters, block widths and relevant floating-point constructor arithmetic |
| Enforce insertion containment and duplicate-safety policy with public provenance; support configurable additional close-pair warnings/errors | Independently enforce containment and fixed safety floor before any native `put()` |
| Reconstruct public IDs, coordinates, face/edge image shifts and diagnostics after native calculation | Check integer overflow in derived Voro++ allocation sizes and enforce a **1 GiB cap on estimated known eager native allocations** |

This separation is based on **information ownership and failure boundaries**, not the rule “simple checks in Python, complicated checks in C++.” pybind11 coercion can erase the original Python type. Conversely, users can import `_core` directly and bypass all public Python checks. Both layers intentionally duplicate some safeguards.

Sources: [public 3D API](../../../src/pyvoro2/api.py#L386-L539), [shared validators](../../../src/pyvoro2/_internal/validation.py), [generator preparation](../../../src/pyvoro2/_internal/generator_preparation.py#L304-L394), [native rectangular preflight](../../../cpp/native_preconditions.hpp#L708-L792), [native periodic preflight](../../../cpp/native_preconditions.hpp#L1731-L1789).

## Backend-unsafe pair policy

The fixed v0.8 threshold is the **binary64 value** `backend_safety_distance_squared = 1e-10`, with nominal distance `backend_safety_distance = 1e-5` in the input coordinate units. Persistent generator pairs (including minimum periodic images) that meet the squared threshold are refused. Native distance classification is conservative: when its computed lower bound cannot establish that a pair is safely beyond the threshold, it may also refuse a near-threshold safe pair.

- Python mandatory scanning runs regardless of `duplicate_check='off'`; an optional user threshold applies **only above** the mandatory safety floor.
- The rectangular native scanner uses a distinct sparse bucket grid, adjacent bucket keys (with periodic wrapping), and conservative binary64 lower bounds.
- The triclinic native scanner uses intervals for the inverse lattice and fractional bucket keys, bounds allowed integer image shifts, and checks lower bounds for candidate shifted distances. It does **not** truncate a required shift search to `[-1,1]`.
- Native cost/resource limits stop certification with a structured error rather than silently ignoring unexamined candidates. Limits include ten million pair comparisons and a one-million budget in specified triclinic enumeration stages.
- `locate` queries are **not inserted**, so they may coincide with a generator; `ghost_cells` queries **are temporarily inserted** and are checked against persistent generators, but not against other ghosts evaluated separately.

Passing this guard means only that this particular proximity hazard was not detected or left uncertified. It is **not** a universal floating-point accuracy certificate, and the threshold is not scale invariant. See [numerical-inputs documentation idea](../documentation-notes/numerical-inputs.md).

## Domain handedness is a public-user-layer contract

In the current v0.8 `PeriodicCell` constructor, original three Cartesian lattice vectors must have a finite **positive** determinant, with additional conditioning limits/warnings ([`domains.py`](../../../src/pyvoro2/domains.py#L281-L340)). The public constructor's transformation to a Voro++ lower-triangular frame assumes this orientation.

Native `preflight_periodic_3d()` does **not** receive the original user lattice. It checks that the internal lower-triangular parameters `bx`, `by`, and `bz` are positive and that their use in the backend is safe. Thus its positive internal determinant does not constitute a second test of the original user handedness.

**Historical context, not current authority:** the abandoned v0.9 attempt implemented accepting both user-lattice handedness signs via an orientation-neutral internal frame ([historical ADR 0018](https://github.com/DeloneCommons/pyvoro2/blob/legacy/v0.9-attempt-2026-10-08/docs/development/decisions/0018-periodic-user-lattice-and-boundary-semantics.md), historical **Phase A / WP2**, not Phase C). It is a possible research reference, **not** an accepted requirement for this reboot.

## Safety is not geometry correctness

- Native preflight checks shapes, arithmetic, insertion validity and known eager memory allocations. Its 1 GiB estimate does **not** bound later dynamic growth, returned Python geometry or peak RSS.
- Finite representable weights/radii can still produce wrong power geometry; numerical gauge cancellation/pruning remain distinct problems ([large-weight investigation](../potential-work/large-weight-numerics.md)).
- Voro++ computes cells independently; success does not certify reciprocal faces, global topology or exact periodic image assignment ([Voro++ contracts](voro.md), [global topology](../potential-work/global-topology.md)).
- Native code still has potential fatal-error paths. The preflight covers identified construction hazards, **not** every backend failure.
- `compute`'s native extraction may report omitted/empty power cells; Python applies its separate result/provenance checks.

## Open review question: face traversal association

[`build_cell_dict()`](../../../cpp/bindings.cpp#L82-L130) calls Voro++ `face_vertices()` and `neighbors()` and pairs results by traversal order. If their *counts* disagree, it retries the two calls in reverse order; if counts still disagree, it raises. The source comment describes rare platform-dependent count inconsistencies.

**Confirmed implementation fact:** this checks **counts**, not semantic face/neighbor correspondence. **Not established:** that an accepted same-count result is wrong, or why reverse call order can repair a mismatch. Before labeling this a defect or changing code, collect a deterministic witness, inspect Voro++ traversal/reset state, and check reciprocal face identity/geometry.

Evidence to consult: [direct-native precondition tests](../../../tests/forward/common/test_native_preconditions.py), [generator-preparation tests](../../../tests/forward/common/test_generator_preparation.py), [native binding source](../../../cpp/bindings.cpp). This note does not claim fresh test execution or an independently proven floating-point certificate.
