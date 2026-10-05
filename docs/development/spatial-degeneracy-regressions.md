# Spatial degeneracy regression dispositions

[Issue #99](https://github.com/DeloneCommons/pyvoro2/issues/99) locks the bounded
spatial decision in [ADR 0025](decisions/0025-native-occurrence-normalization-and-proof-assisted-identities.md).
The regressions do not accept #95 or Checkpoint B. Spatial normalization remains
a numerical organization of raw local representations. Exact scientific
point/ridge/face incidence belongs to WP5's separate complete exact audit.
Strict normalized success is an error action over enabled/applicable numerical
checks, not exact-S reconstruction, E=S topology or an N-to-S vertex bijection.
Unavailable exact membership alone does not invalidate that weaker raw view.

## Evidence and independent oracles

Inputs are selected from the accepted #96 atlas without changing their sites,
weights, radii, bases, IDs or periodicity. The input-only file
`tests/forward/spatial/data/degeneracy_scope.json` has 33 named cases; it contains
no archived native output or normalization golden dataset. Expected exact facts
are recomputed by `_degeneracy_oracle.py`, a compact extraction of #96's
independent Fraction polygon-clipping oracle. Self-image/wall outer bounds and
outward rational roots establish complete finite image enumeration. Exact
closed-boundary and Euler checks protect that oracle. It imports no production
geometry or normalizer. Production WP5 vertices/contact dimensions are checked
separately in doubled source-local coordinates.

The analytic Cartesian derivation gives n vertex classes, 3n edge classes and
3n face classes for n cuboidal owners; each local cell has 8V/12E/6F. For the
oblique prism, bisectors `x=±1/2` and `±x/2±y=5/8` give the six exact xy corners
`(±1/2,±3/8)` and `(0,±5/8)`. Its z extrusion has 12V/18E/8F locally and
2V/5E/4F/1C in the quotient. These analytic checks supplement the clipping
oracle. Neither derivation supplies native vertex membership.

Verified inherited evidence identities:

| File | SHA-256 |
|---|---|
| `issue96-degeneracy-atlas-ba9e58f.md` | `67c349f2eca906feecf730a85a9b04735ce1c108a452eb117c4ce1ab504992b9` |
| `issue96-degeneracy-atlas-ba9e58f.zip` | `2c9aa41ca48c6a7a32fa76bd66765b4b5afcc75bcc2cd27797871a186b63f119` |
| atlas `SHA256SUMS` (1,264 entries) | `da8e40eb5747de1b0ca56a2c34837a983979fb8fd8ad48224ab28eaaa984e829` |
| `checkpoint-b-degeneracy-contract-ba9e58f.md` | `c3623cf93f1aa5aefb2cdcc8f551b1cc2c8262be45ef4eeac5470dd19dc02c9b` |
| `checkpoint-b-degeneracy-review-verification-ba9e58f.zip` | `e2ee7bf339dfd42ed757b3a0e624416f0bef1bb3a2357e7f4a41ba8c98aa449a` |

The atlas remains historical evidence. ADR 0025 and current source-controlled
contracts remain normative. Historical native results are not regenerated to
fit this implementation.

## Durable case coverage

All sixteen spatial base archetypes are represented. The table names the
corresponding tests in `test_degeneracy_scope.py`. Counts are exact S V/E/F/C;
they are not required normalized raw counts.

| Archived archetypes | Exact fact / protected boundary | Test |
|---|---|---|
| `spatial_self_cube`, `spatial_mixed_slab8`, `spatial_mixed_square_extruded8`, `spatial_distinct_cube8` | 1/3/3/1, 2/6/6/2, 4/12/12/4, 8/24/24/8; same cuboidal local geometry, eight image-qualified lifts | `test_cartesian_local_geometry_does_not_determine_owner_quotient` |
| `spatial_self_hexprism6` | 2/5/4/1; six-way target; exact ridges and faces retain their different dimensions | `test_hexagonal_prism_has_six_way_incidence_without_point_collapse` |
| `spatial_partial_square_ridge4` | 8/20/16/4; endpoints (1/2,1/2,0) and (1/2,1/2,1) are distinct because z is nonperiodic | `test_partial_periodic_ridge_retains_two_distinct_quotient_endpoints` |
| `spatial_asymmetric_power5` and `__exact_dyadic_radii` | Successful complete audit coexists with E=29/59/35/5 and S=28/58/35/5; direct radii give E=S | `test_successful_power_five_audit_is_not_e_s_vertex_bijection` |
| `spatial_asymmetric_power6__exact_dyadic_radii` | E=S=17/40/29/6, including exact contacts | `test_power_six_direct_dyadic_radius_control_has_identical_e_s_geometry` |
| `spatial_generic_tetra4`, `spatial_bipyramid5`, `spatial_octa6`, `spatial_asymmetric_power6` | 12/32/24/4, 11/30/24/5, 2/11/15/6, 17/40/29/6 | `test_other_spatial_archetypes_keep_exact_incidence_and_raw_scope` |
| `spatial_self_triclinic4`, `spatial_partial_cube8`, `spatial_rotated_self_cube8`, `spatial_asymmetric_power8`, `spatial_asymmetric_extruded_power4` | 6/12/7/1, 12/32/28/8, 1/3/3/1, 44/92/56/8, 6/16/14/4 | Same test; exact vertices checked independently |
| `spatial_bipyramid5__radial_out_2m36` | Independently positive S facet missing from N; WP5 coverage finding and raise action survive; numerical strict refusal survives | `test_tiny_exact_positive_facet_loss_remains_wp5_coverage_failure` |
| `spatial_distinct_cube8__radial_in_2m52` | Three pairs of distinct exact S vertices round to identical public triples; existing WP5 coverage refusal remains separate from strict raw success | `test_equal_public_triples_do_not_identify_exact_semantic_vertices` |

All seven saved strict-passing raw/S-count disagreements are retained by
`test_all_seven_saved_strict_views_may_differ_from_s`. They were identified from
the atlas's `summary.json`, core index and supplementary search index:

| Case | Exact S V/E/F/C | Accepted audit outcome |
|---|---|---|
| `spatial_bipyramid5__radial_out_2m24` | 15/36/26/5 | Missing positive coverage |
| `spatial_bipyramid5__radial_in_2m24` | 16/37/26/5 | Complete successful audit |
| `spatial_octa6__radial_in_2m24` | 9/23/20/6 | Missing positive coverage |
| `spatial_octa6__tangential_2m24` | 9/23/20/6 | Complete successful audit |
| `spatial_asymmetric_power5__radial_out_2m36` | 30/61/36/5 | Missing positive coverage |
| `spatial_asymmetric_power5__tangential_2m36` | 30/61/36/5 | Missing positive coverage |
| `spatial_distinct_cube8__radial_in_2m52` | 26/64/46/8 | Missing positive coverage |

Strict success in the first, third and last three rows does not erase the
owning WP5 error. A consumer requiring successful exact scientific checking
must still refuse. No face is synthesized and no diagnostic is downgraded.

Seven additional input controls cover self/distinct equivalent shears,
self/distinct left-handed bases, a left-handed equivalent hexagonal prism,
sparse cube IDs and independent self-image translations. Each runs in standard
and equal-radius power modes and checks both normalized local reconstruction
and WP8 user-basis query/owner shift equations. Backend-primary representatives
are preserved; user-cell wrapping is not substituted for normalization.
`test_spatial_routes_do_not_consume_planar_proof_adapter` protects the dimension
boundary. The existing #98 tests protect the 4V/12 raw-edge/20-occurrence square,
45/45 and 105/105 central closures, conservative 11V/17-edge standalone square,
and loss of proof authority on copying/serialization.

## Spatial failure and relation dispositions

The four dispositions below apply independently to the named layer/operation.
A case can have a supported numerical view and a failed exact coverage audit;
these outcomes must not be conflated.

| Archived spatial family / observation | v0.9 disposition |
|---|---|
| All sixteen base archetypes, supported order/ID/translation/equal-power/handedness/basis controls; 100 archived strict-passing numerical views, plus four successful cosphere pilots and the rotated distinct-cube pilot | **Supported numerical raw view.** Preserve local records, owner/image labels, mappings and applicable consistency checks; no exact-S count promise. |
| Seven saved strict-pass count disagreements, including the rounding case | **Supported numerical raw view.** Explicitly tested; exact audit findings remain independently active. |
| Positive-facet loss in small bipyramid/octahedron/cube/square-extrusion/asymmetric-power perturbations, seeded power8/extruded-power4 pilots and the rotated distinct-cube pilot | **Existing exact semantic/audit finding or refusal.** Preserve `WP5_POSITIVE_FACET_MISSING` in each named E/S ideal and `WP5_RECIPROCAL_MISSING` where observed; diagnose/warn/raise actions remain unchanged. |
| Twelve archived strict failures: bipyramid radial-out/radial-in/tangential 2^-36; octahedral radial-out 2^-24 and radial-out/radial-in/tangential 2^-36; cube radial-out/radial-in/tangential 2^-36; square-extrusion radial-out/tangential 2^-36 | **Numerical representation failure/limitation.** Keep image-qualified shift/set/incidence findings; the bipyramid representative is durable. No inputs are perturbed again to manufacture a pass. |
| Successful asymmetric power-five E/S V/E split; inexact-radius/weight bridge; frame/orthogonal-pilot E/S vertex bridges | **Stronger exact projection intentionally out of v0.9 scope.** Audit success can stand; E=S and a vertex bijection are separate, unprovided claims. Status conflicts, if encountered, retain WP5 conflict/refusal authority. |
| Support-intersection candidates, normalized candidate conflicts and incomplete N-to-S membership, including 57 unproved cube slots | **Stronger exact projection intentionally out of v0.9 scope.** Candidate conflicts are not contradictions between proved S identities. Singleton support intersections supply no identity permission. |
| Direction-dependent generic perturbation resolutions | **Stronger exact projection intentionally out of v0.9 scope.** Each fixed S input has an exact complex; a sampled resolution does not define perturbation-independent degeneracy policy. |
| Distinct exact vertices sharing a rounded coordinate triple | **Existing exact semantic/audit finding or refusal** for the physical coverage counterexample. A coordinate-only exact projection is **intentionally out of scope**. Numerical coincidence is not semantic identity. |
| Missing or malformed normalized mappings/indices, source ownership, nonperiodic shifts, aligned edge/face classes | **Numerical representation failure/limitation.** Missing consumed data is an error; it cannot be reported as checked success. |
| Coordinate-coincident differently attributed or transitively linked schema cycles | **Numerical representation failure/limitation.** Preserve the existing standalone ambiguity guards; these dictionary controls are not physical producer observations. |
| Hypothetical retained lower-dimensional N face and the finite negative search | **Existing exact semantic/audit finding or refusal.** Existing WP5 zero/absent/cycle/conflict/coverage semantics govern; any required new collapse policy returns to #95 contract review. |
| Missing component qualification, unsupported route or hostile runtime state | **Existing exact semantic/audit finding or refusal.** Admission remains atomic; geometry or topology success cannot grant artifact qualification. |

The atlas's seventeen spatial relation names have these bounded meanings:

| Relations | Disposition and authority boundary |
|---|---|
| `exact_facet_vertex_incidence`, `exact_ridge_endpoint_incidence`, `exact_point_stratum_vertex_incidence` | **Existing exact semantic/audit facts.** Exact dimensions and endpoints are meaningful; incidence supplies no normalized vertex identity. |
| `contradicted_merge_distinct_ridge_endpoints` | **Existing exact semantic/audit fact.** Distinct lifted endpoints forbid merge-all; partial-periodic endpoints also differ in the quotient. |
| `exact_image_transport` | **Existing exact semantic/audit fact.** Ideal S lattice orbits remain within S; no automatic native-slot projection. |
| `certificate_label_attribution`, `native_cycle_vertex_incidence`, `native_edge_endpoint` | **Supported numerical raw view** with WP5's existing provenance authority. Preserve source/image/token/slot/cycle association and occurrence multiplicity. |
| `exact_native_ideal_vertex_membership`, `exact_E_S_point_bridge`, `exact_native_image_transport`, `exact_native_edge_ridge_geometry` | **Stronger exact projection intentionally out of v0.9 scope.** Keep these independently proved partial facts in their named domains; no new spatial identity consumer or universal closure follows. |
| `positive_reciprocal_correspondence` | **Existing exact semantic/audit fact** and applicable numerical compatibility. Repeated positive raw classes compare image-qualified class unions, not arbitrary one-to-one fragment pairings. |
| `unproved_support_incidence_candidate`, `numerical_only_coincidence`, `unproved_candidate_relation`, `unproved_E_S_vertex_bridge` | **Stronger exact projection intentionally out of v0.9 scope.** Unavailable/candidate relations remain unproved; numerical views acquire no certification from them. |

No retained lower-dimensional **native N face** was observed in the finite #96
campaign. The search covered 5,634 packet faces plus 304 additional pilot faces.
This is **not an impossibility theorem**. Exact E/S ridges and points were
observed and remain meaningful. No physical retained-zero-N-face fixture is
fabricated. Existing WP5 synthetic zero/cycle/coverage controls remain labeled
as schema tests. `test_normalized_schema.py` adds representation mutations and
an explicitly synthetic positive rectangle partition to protect class-union
reciprocity; it claims no new Voro++ producer behavior.

An enabled applicable check requires its face-cycle operands to be present and
non-null; an available empty cycle is distinct from missing input. Euler-only
`NormalizedVertices` validation consumes local cycles without requiring unused
neighbor/image shifts. Enabled periodic face consumers still require those
shifts, and full topology views retain their stronger mapping obligations.
Missing consumed operands use existing error diagnostics in basic mode and
raise `NormalizationError` in strict mode. Euler warnings retain their severity
and counts when examples are truncated.

## Bounded omissions

No required base archetype or saved strict-pass disagreement is omitted. The
remaining 84 physical archived inputs are intentionally not permanent fixtures:
52 duplicated core order/translation/equal-power/bridge variants and additional
perturbation directions/scales, 15 additional power8 seeds, 7 additional
extruded-power4 seeds, 5 orthogonal-frame pilots and 5 root search pilots repeat
the dispositions above. Seed-zero bases are promoted; existing WP5/WP8 tests
also cover frame/source transport. The complete archived negative evidence
remains unchanged. Two spatial dictionary controls are
represented by the existing direct/transitive ambiguity tests rather than
copied as native fixtures. The handoff enumerates every unpromoted case and its
reason; this page groups them by mechanism to keep the contract readable.

The implementation adds no spatial certificate-aware adapter, dimension-blind
zero merge, ridge/face collapse engine, generic exact complex, proof callback,
native/schema extension or public API. WP7 ghost-collapse policy remains owned
by its separate route. Changed measured closures require explicit mechanical
manifest refresh and fresh technical artifact qualification under ADR 0024.
Independent final review follows validation of the complete candidate.
