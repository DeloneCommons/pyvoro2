# 0034 — Stable source lifting for native generator preparation

- **Status:** Proposed — implementation review pending
- **Date:** 2026-11-28
- **Scope:** [F1-A3-P #166](https://github.com/DeloneCommons/pyvoro2/issues/166)
- **Related:** [A1](0030-exact-user-lattice-and-native-admission.md),
  [A2](0031-qualified-native-periodic-qr-frame.md),
  [R](0032-native-primary-remapping.md),
  [S](0033-source-connectors-and-realization-reference.md),
  [mandatory safety](0013-central-generator-preparation-and-backend-safety.md)
- **Narrow supersession proposed:** ADR 0032's retained ordinary Cartesian
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

S retains original-source connectors, source identity, raw contact labels and
the same-operation v/K/L/anchor reference with its existing membership/refusal
policy. No inverse state or active-set algorithm changes. Uninserted locate
queries keep direct projection/dispatch and reporting; safe remote stepping and
optional query-chart recovery remain F2. No release or F1 closure is declared.

## Alternatives and validation

Projecting the remote input retains the demonstrated phase loss. Rounded user
wrapping/residuals, floating inverse/floor, BLAS dots and reconstructing remote
`k@L` do not implement the selected exact-operand expression. General numerical
certification, new public abstractions and speculative fast paths are unnecessary.

Review uses independent Fraction solve/dot/chart oracles, actual compiled R1
submissions and analytic per-ID measures, S composition, both safety witnesses,
bounded query preservation, unmodified record-level P0 replay and measured
integrated preparation/compute costs. Acceptance must be recorded in this same
implementation PR after independent review; this record remains Proposed.
