# Local projection rounding before native reduction

**Status:** R-L03a follow-up, pending a separate maintainer decision. F1-A and
[ADR 0034](../../../docs/development/decisions/0034-stable-source-lifting.md)
are accepted within their stated limits. This note preserves independent audit
findings and a candidate refinement, not a selected algorithm, implementation
issue, priority, release gate or B1 dependency.

## Provenance and disposition

The independent final F1-A acceptance audit (2026-11-28) examined
[PR #167](https://github.com/DeloneCommons/pyvoro2/pull/167) at
`2fa9cd380c097da1cc6be61318faa2f5c6fecbc8`, against integrated base
`07a619b0a723ff860ea577a07489d0c21fefb01b`. It independently reproduced both the motivating remote-lift
defect and newly introduced local-resolution loss. The accepted P policy
restores exact-translation covariance, with no demonstrated Blocking or
Important production defect or material new wrong successful ordinary-use
tessellation in the bounded audit. It does not preserve every source detail.

The observations below are extracted from that audit, not a new experiment
performed during documentation closure. The independent evidence includes a
fresh GCC 14.2/C++17 Release build with CPython 3.12.14 and NumPy 2.3.5.
See the pinned [coordinator](https://github.com/DeloneCommons/pyvoro2/blob/2fa9cd380c097da1cc6be61318faa2f5c6fecbc8/src/pyvoro2/_internal/native_periodic_preparation.py),
[reducer](https://github.com/DeloneCommons/pyvoro2/blob/2fa9cd380c097da1cc6be61318faa2f5c6fecbc8/src/pyvoro2/_internal/native_remap.py)
and [implementation evidence](https://github.com/DeloneCommons/pyvoro2/blob/2fa9cd380c097da1cc6be61318faa2f5c6fecbc8/notes/maintainer-review/contracts/f1-a3-p/README.md).

## Mechanism: fractional locality is not Cartesian locality

P computes exact k=floor((p-o)@inverse(A)), r=p-o-k@A, then
u=RN(r@Q), before the existing R reducer. For theta=(p-o)@inverse(A)-k,
r=theta@A, with theta components in [0,1). Therefore, for the exact
projected residual s=r@Q,

```text
abs(s_j) <= sum_i abs((A@Q)_ij)
```

where the right-hand expression is interpreted on exact represented operands.
For normal values the additional rounding error is bounded by
2**-53 times abs(s_j); subnormals have an absolute rounding bound. The
scale is that of the basis, not necessarily the shortest physical period.
R cannot recover phase lost at this intermediate rounding.

The identity-frame dyadic witness uses A=((1,0,0),(2**20,1,0),(0,0,1)),
o=0 and p=(2**-35,1/2,1/2). A2 admits it with a conditioning warning
(condition number about 2**40). Q=I and L=A exactly, ignoring signed zeros.
The exact k=(-2**19,0,0) gives r_x=2**19+2**-35. Binary64 spacing at
2**19 is 2**-33, so u_x=2**19. R commits n=(2**19,0,0), yielding
v=(0,1/2,1/2) and K=(0,0,0). The pre-P direct projection retained
2**-35, including at the existing default epsilon 1e-12; P loses it
even at eps=0. This is distinct from QR error, snapping and caller-lost bits.

Safe Python-level reproducer (no native container construction):

```python
from pyvoro2 import PeriodicCell

cell = PeriodicCell(((1., 0., 0.), (2.**20, 1., 0.), (0., 0., 1.)))
p = [[2.**-35, .5, .5]]
direct = cell.remap_internal(cell.cart_to_internal(p), eps=0.)
primary, shifts = cell.remap_cart(p, eps=0., return_shifts=True)
# direct[0, 0] == 2**-35; primary[0, 0] == 0; shifts == [[0, 0, 0]].
# The existing conditioning warning is expected.
```

## Observed magnitude and consequences

Independent identity-frame probes (first row (1,0,0), third row (0,0,1)):

| Second row | Approx. condition number | A2 warning | Lost x component |
| --- | ---: | :---: | ---: |
| (1,1,0) | 2.618 | No | 2**-54 |
| (16,1,0) | 258 | No | 2**-51 |
| (256,1,0) | 6.55e4 | No | 2**-47 |
| (2**16,1,0) | 4.29e9 | No | 2**-39 |
| (2**20,1,0) | 1.10e12 | Yes | 2**-35 |
| (2**24,1,0) | 2.81e14 | Yes | 2**-31 |
| (2**31,2**31,0) | 4.29e9 | No | 2**-24 |
| (2**43,2**37,0) | 5.63e14 | Yes | 2**-12 |
| (2**48,2**48,0), eps=0 | 5.63e14 | Yes | 2**-7 |

The table records audited helper probes, not a prevalence estimate or
tessellation error bound. The last case refuses default epsilon overlap.
The preceding anisotropic case admits default epsilon 0.137438953472.

For A=((1,0,0),(2**43,2**37,0),(0,0,1)),
p0=(3/8,2**36,1/2) and p1=(3/8+2**-12,2**36,1/2), the exact
source-periodic separation is 1/4096, more than 24 times the nominal
backend safety distance 1e-5. P maps both x coordinates to 3/8.
The actual public duplicate diagnostic returns zero distance:

```python
from pyvoro2 import PeriodicCell, duplicate_check

cell = PeriodicCell(((1., 0., 0.), (2.**43, 2.**37, 0.), (0., 0., 1.)))
points = [[.375, 2.**36, .5], [.375 + 2.**-12, 2.**36, .5]]
pairs = duplicate_check(points, domain=cell, threshold=1e-5, wrap=True,
                        mode='return')
# One pair at distance zero; expected conditioning warning.
```

This is correct for the prepared Cartesian operands, not for original-source
separation. The mandatory prepared/native insertion safety policies refuse
such collisions. A separate moderate-shear-8 diagnostic witness crossed the
threshold from source separation 1.0000000000037756e-5 to prepared separation
9.999999999621423e-6 (coordinate change about 4.16e-16). Small local errors
can therefore affect discrete near-boundary decisions even at moderate shear.
For an analytic three-site aligned unit-lattice control, losing the original
2**-35 phase changes two slab volumes by 2**-36. This is a perturbation, not
evidence of an order-one error in an executed extreme tessellation.

## Native feasibility is a separate question

The audit inspected preflight and used its checked resource-estimate hook,
not extreme native execution:

- Unit-diagonal shear 2**20 fails the extended-block product even at the
  minimum legal block counts.
- Unit-diagonal shear 4096 estimates about 3.30 GB at minimum blocks, beyond
  the existing 1 GiB cap.
- The large anisotropic examples above with z period 1 fail a checked extent.
- Changing z period to 2**37 in the (2**43,2**37,0) example allows an estimate
  of 1,089,277 bytes at minimum blocks, although the default two-point grid
  still refuses.

Resource checks therefore do not exclude every potentially relevant case.
Passing a resource estimate is not proof of accurate native geometry.
Do not execute unsafe or unqualified extreme container/query arithmetic as a
regression strategy. A separate shear-8 face-attribution refusal in the audit
was reproduced on the pre-P base and belongs to the existing G1a attribution
limitations, not to this local-projection mechanism.

## Candidate refinement: retain exact projection through reduction

If justified by a named workload, compare the accepted path

```text
r exact -> u=RN(r@Q) -> R(u,L,eps)
```

with passing the exact dyadic projection r@Q (integer numerators over a
power-of-two denominator) into a narrow shared integer reduction entry, then
rounding the emitted coordinates according to the selected R chart policy.

Algebraically this removes the intermediate loss in the displayed identity-frame
witnesses and preserves exact source-translation covariance; it needs neither B1
reduction nor a second numerical subsystem. It is **not** an equivalent
optimization: exact operands can change quotient/endpoint decisions relative to
the currently represented u. The rounding boundary in ADR 0034 would change.
Specify that revised contract explicitly before implementation. Preserve the
ordinary binary64-input internal remap contract, full-row committed shears,
rounded-upper advancement, snapping/nonoverlap policy, wide combined K,
A1/A2/S contracts and both safety stages.

A centered fractional representative only reduces the residual-size bound by
a bounded factor. A shortcut based on original point magnitude can destroy
translation covariance. Neither a different chart nor a general certification
framework is selected.

Before any scoped implementation: reproduce the exact helper witnesses, add
independent rational/chart and nearest-even boundary oracles, assess real
well-scaled and sheared/anisotropic workloads (including near-threshold
duplicates and owners where native admission permits), compare changed
endpoint/shift decisions and cost, preserve P0 and R1/S integration. Distinguish
prepared-view and original-source diagnostics. Do not silently relax the
duplicate floor or suggest wrap=False as an exact periodic-source workaround.
The refinement remains conditional; F1-A acceptance and separately scoped
F2/B/C/G1 work do not depend on it.
