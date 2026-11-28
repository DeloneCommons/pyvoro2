# 0034 — Stable source lifting for native generator preparation

- **Status:** Accepted — maintainer-approved after independent A3-P and
  F1-A composition audit and documentation corrections
- **Date:** 2026-11-28
- **Scope:** [F1-A3-P #166](https://github.com/DeloneCommons/pyvoro2/issues/166)
- **Implementation:** [PR #167](https://github.com/DeloneCommons/pyvoro2/pull/167)
- **Independently reviewed HEAD:** `2fa9cd380c097da1cc6be61318faa2f5c6fecbc8`;
  [six required CI jobs successful on the reviewed implementation](https://github.com/DeloneCommons/pyvoro2/actions/runs/46096680769)
- **Related:** [A1](0030-exact-user-lattice-and-native-admission.md),
  [A2](0031-qualified-native-periodic-qr-frame.md),
  [R](0032-native-primary-remapping.md),
  [S](0033-source-connectors-and-realization-reference.md),
  [mandatory safety](0013-central-generator-preparation-and-backend-safety.md)
- **Narrow supersession:** ADR 0032's retained ordinary Cartesian
  projection in composite remapping, not its reducer or endpoint policy

## Context

Projecting a remote source point before reduction can lose its local phase.
R's exact reduction cannot recover that information. On the unimodular R1
lattice, translating only the middle site by an exactly represented `2**50`
copies of the first source row changes per-ID volumes, middle-site ownership
and singleton ghost volume on the integrated baseline. No new frame admission
threshold or basis reduction is needed for this witness.

## Selected policy

For canonical finite binary64 p, original row basis A, origin o and the actual
paired operation Q/L, compute:

```text
f = exact (p-o) @ inverse(A)
k = floor(f)
r = exact (p-o-k@A)
u = RN(exact r@Q)                # nearest-even, once per complete component
(v,n) = shared native remap(u,L,eps)
K = k+n
```

Represented source and Q coefficients are exact operands. Keep the local
residual in integer/dyadic form through projection, sharing A1's bounded
immutable basis/cofactor data. Keep k and n as Python integers and check only
K at the legacy int64 destination, including unreturned public shifts and
`PreparedGenerators.remap_shifts`. Nonfinite views or final overflow refuse
without overflow/invalid-cast warnings. A1's different optional-shift contract
is unchanged.

One private coordinator serves public/snapshot Cartesian remap, persistent
generators for spatial compute/locate/ghosts, and temporary inserted ghosts.
It consumes the existing frame without recomputation or persistent caching.
R's represented operands, epsilon/nonoverlap policy and endpoint decisions,
A2 admission/parity, rectangular routes, standalone converters/internal remap
and the four exact A1 helpers are unchanged. Cartesian reports use the existing
native-to-Cartesian transform and finite checks; owner NaN reporting remains.

For exactly represented p'=p+t@A, exact floors obey k'=k+t and r'=r. Thus, with
the same Q/L and epsilon, native submissions are identical and K'=K+t when both
operations are otherwise admitted, independent of batch shape/order. This does
not recover bits lost before input or promise equality between different frames
or basis representations.

## Consequences and limits

`remap_cart` need not equal a literal composition of separately rounded public
helpers. K is a nominal preparation coefficient; local projection, snapping
and reporting still round. Neither exact source/native equivalence, exact
Cartesian reconstruction, arbitrary short-vector accuracy nor source-complete
contacts follows. Both prepared Cartesian duplicate/containment checks and the
actual native insertion safety stage remain mandatory; distinct source points
can still refuse when reporting collapses them or native coordinates are unsafe.

Here "local" refers to fractional coordinates in the original parallelepiped.
A strongly sheared or anisotropic basis can still produce a large Cartesian
residual. Rounding its projection before native reduction can discard phase
present in the original point, including information retained by the former
direct projection. This occurs even for `Q=I`, `L=A` and `eps=0`. For rows
`(1,0,0)`, `(2**20,1,0)`, `(0,0,1)`, origin zero and
`p=(2**-35,1/2,1/2)`, exact lifting gives `k=(-2**19,0,0)` and
`r_x=2**19+2**-35`. The projection rounds x to `2**19`; reduction emits x=0
and final `K=(0,0,0)`. The earlier direct projection retained `2**-35`.
Prepared-coordinate duplicate decisions can change, including collapse of
source points separated by more than the backend safety distance; the
mandatory checks conservatively refuse such collisions. Translation covariance
is invariance of this selected numerical preparation, not preservation of every
local source detail. Frame admission and warnings are not uniform accuracy
certificates. See the [domain guide](../../guide/domains.md#backend-primary-remapping-helpers)
and [duplicate check reference](../../reference/duplicates.md).

S retains original-source connectors, source identity, raw contact labels and
the same-operation v/K/L/anchor reference with its existing membership/refusal
policy. No inverse state or active-set algorithm changes. Uninserted locate
queries keep direct projection/dispatch and reporting; safe remote stepping and
optional query-chart recovery remain F2. This decision closes the selected
F1-A scope only; it does not close F1-B/C, qualify all forward operations,
establish source-complete contacts or approve a release.

## Alternatives and validation

Projecting the remote input retains the demonstrated phase loss. Rounded user
wrapping/residuals, floating inverse/floor, BLAS dots and reconstructing remote
`k@L` do not implement the selected exact-operand expression. General numerical
certification, new public abstractions and speculative fast paths are unnecessary.

Review uses independent solve/dot/chart oracles, actual compiled R1 submissions
and analytic per-ID measures, S composition, both safety witnesses, bounded
query preservation, unmodified record-level P0 replay and measured integrated
preparation/compute costs. See the [implementation evidence at the reviewed HEAD](https://github.com/DeloneCommons/pyvoro2/blob/2fa9cd380c097da1cc6be61318faa2f5c6fecbc8/notes/maintainer-review/contracts/f1-a3-p/README.md).

The independent final F1-A audit (2026-11-28) found no Blocking or Important
production finding. On a fresh current native build it passed 334 disjoint
repository tests with warnings treated as errors, 558 source solve/residual
comparisons, 1,674 projection/chart comparisons and 186 exact translation pairs.
Independent compiled controls recovered R1 standard/power per-ID volumes,
unique owners and standard/power singleton ghost measures. A separate oblique
lattice was tested in 36 compiled matching runs with signed/permuted bases,
individual translations and three modes; inverse label transport, active-state
changes and both mandatory safety stages remained coherent. Both base and
candidate P0 replays passed all 48 complete, identical records. Required CI on
the audited implementation also succeeded. The two audit findings required
only documentation corrections: the local-resolution limitation above and the
scope of the reducer's rounding claim. The audit did not execute extreme native
containers or unsafe remote queries and did not rerun a comprehensive timing or
platform campaign.

The maintainer accepts A3-P and the bounded A1/A2/R/S/P composition in this PR.
The independently confirmed local-resolution loss is retained as a concrete
future numerical refinement candidate under R-L03a in the maintainer-review
register, not an accepted algorithm change or a condition of this closure.
