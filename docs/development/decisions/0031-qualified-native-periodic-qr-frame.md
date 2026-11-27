# 0031 — Qualified native periodic QR frame with both handedness signs

- **Status:** Proposed — implementation authorized by #159; acceptance pending independent review
- **Date:** 2026-11-27
- **Issue:** [F1-A2 #159](https://github.com/DeloneCommons/pyvoro2/issues/159)
- **Narrow supersession:** ADR 0030's retained native right-handedness condition only
- **Related:** [ADR 0030](0030-exact-user-lattice-and-native-admission.md),
  [ADR 0012](0012-certified-periodic-image-geometry.md),
  [ADR 0013](0013-central-generator-preparation-and-backend-safety.md)

## Context

The two Gram–Schmidt native frames lose isometry for near-parallel rows.
For rows `(1,1,0)`, `(1,1+2^-30,0)`, `(0,0,1)`, P0 records an orthogonality
error about `4.77e-7` and a native/source volume ratio about 1025. The exact
A1 lattice remains valid independently of numerical native feasibility.

## Decision

Use one private, cycle-free frame preparation for public backend helpers and
detached native snapshots. With the unchanged source row matrix `A`, compute
nonpivoted NumPy Householder QR of **the transpose**:

```text
Q0, R0 = qr(A.T)
D = diag(sign(diag(R0)))
Q = Q0 @ D; R = D @ R0; L = R.T; C = Q.T
A.T ≈ Q @ R; A @ Q ≈ L
y = (x-o) @ Q; x = o + y @ Q.T
(bx,bxy,by,bxz,byz,bz) = (R00,R01,R11,R02,R12,R22)
```

Check factor finiteness and nonzero diagonals before normalization. Require
positive triangular diagonals and a finite positive computed native volume.
The exact binary64 source determinant sign governs orientation; independently
check the computed transform parity. For `u=2^-53`, require
`||C C.T-I||_inf <= 60u` and
`max_i sum_j |L_ij-(A Q)_ij| / sum_j |A_ij| <= 60u`.
Failures raise native-adapter `ValueError`, without declaring the source
mathematically singular. Material redesign or relaxation of these gates needs
maintainer approval and an independently reproduced witness.

Retain finite positive row norms and their product, a finite nonzero numerical
determinant with the exact source sign, successful positive finite SVD values,
`relvol >= 1e-12`, `cond <= 1e15`, and the warning above `1e10` at native use.
Native preflight, duplicate, containment and resource guards still apply.
A forward operation prepares one detached frame with owned readonly arrays and
immutable parameters, reusing it for coordinates, parameters and reporting.

Allow both source signs: `det(Q)` has the source sign while `det(L)>0`.
After native face-shift inference, a shared reporting helper transforms sites
and vertices and reverses each present face cycle and cyclic vertex adjacency
once for improper frames, including output without vertices. This preserves
Voro++'s established **inward** face-cycle convention and negative oriented
volume, while reported scalar volumes remain positive. IDs, vertex indices,
face order and neighbor/image pairings remain unchanged. Locate only transforms
owner positions, including the established missing-owner sentinel.

Accept either exact nonzero determinant and all eight nonzero signed diagonal
patterns in the exact image kernel **only for fully periodic 3D**. Exact signed
floor/ceiling candidates retain source coefficients, distance keys, pair
reversal, coefficient-space ties, completeness, budgets and range failures.
Planar and partial-periodic admission stays unchanged. Mandatory source-exact
duplicate scans continue to use `A`, never the numerical triangular factor.

### Narrow R1 remapping prerequisite

The independent Pro audit of PR #160 found that valid QR rounding can activate
a destructive pre-existing coupled upper snap. For the unimodular rows
`(0,-1,-1)`, `(1,0,0)`, `(-1,-1,0)`, a tiny negative native z residual triggers
ordinary reduction and snapping; the ensuing y overshoot contains a genuine
quarter-unit remainder. A lower-bound-only snap discards that remainder.
Three ordinary source sites then receive wrong ID-associated volumes, even
though the resulting cells pass tessellation diagnostics.

The maintainer explicitly authorized this narrow A3 prerequisite inside A2:
bound the y and x upper-snap predicates on both sides, `b-eps <= v <= b+eps`.
Larger coupled overshoots retain their remainder and undergo the existing
ordinary lattice reduction. QR, its `60u` gates, epsilon defaults, `eps=0`,
half-open conventions, source-row shifts and native safety remain unchanged.
This is an exception to the original A3 exclusion and unchanged-nonframe P0
requirement, necessary for accepting A2; it is not general A3 implementation.

The unmodified P0 replay changes exactly seven of 48 records: the #65 public,
snapshot and compiled records retain `x=3/16`; the four #67 B-z-to-y-to-x
public/snapshot records with default or explicit `eps=1e-12` also retain that
tangential remainder. Their original oracles now pass. The other 41 records,
including #60, all `eps=0` cases and all frame records, are unchanged. Eighteen
records still report discrepancies; source snapping, zero-epsilon endpoints,
subnormal behavior and broader remapping correctness remain deferred to A3.

## Consequences and limits

The source lattice, origin, row order, four exact A1 operations and positive
`from_params` convention remain unchanged. A native image uses the same integer
coefficient `s` as a source image, but `L ≈ A @ Q` is not an exact binary64
identity. Matching integer labels does not certify Cartesian translations,
insertion provenance or source-complete geometry.

QR removes the P0 gross frame defect; it does not guarantee small relative
error in every short translation (including the `2^-38` witness) or a correct
native tessellation. Beyond the narrow R1 prerequisite above, A3 source
snapping/coupled remapping/endpoints, B1 reduction,
C1 physical ties, C4 translation recovery, G1 contacts/complexes and F2 query
interfaces remain separate. No source rebasing, automatic rescaling, vendor
patch, dependency, persistent context or radius cutoff is introduced. The
[power-radius limitation](../../guide/operations.md) remains.

## Alternatives and acceptance gate

Retaining duplicated Gram–Schmidt leaves a demonstrated nonisometry. Recomputing
parameters from transformed source vectors would abandon the paired QR factor.
Pivoting, source sign changes, basis reduction and automatic scaling change
other contracts and are outside A2. Universal short-vector relative-error
cutoffs are not justified by a backward-error qualification.

Acceptance requires independent mathematical, orientation and native-safety
review of the implementation and final-HEAD CI. Tests include rational and
analytic frame/image oracles, compiled right/left pairs, signed duplicate
buckets, numerical fault injection, ownership/lifetime, deterministic dyadic
coupled-snap reconstruction, source-analytic R1 compute/locate/ghost witnesses,
and individual P0 reconciliation under the authorized exception above.
This Proposed record does not mark A2 or a release Accepted.
