# Functional coverage — follow-up review checklist

**Status:** working questions for a systematic second pass over pyvoro2 functionality. This is **not** an implementation plan, an assertion of complete coverage, an API decision, or a v0.9 release gate. The [review register](review-register.md) remains the single decision/status index.

## Sequencing and evidence

- **Do not block high-priority development on this review.** It can be done incrementally, including *after* work on point/edge identity and globalization (R-O01, P-01a) and the individually configured inverse rows/terms (R-I01–R-I03), once each has its own required design approval. These examples are priorities for discussion, not automatic implementation authorizations.
- For each topic, contrast actual v0.8-based source/API behavior, tests, known historical counterexamples and documented contracts. Do not assume a historical v0.9 fix was imported by the reboot. Prefer a small representative route or test over a wholesale line-by-line audit.
- Mark a question here as inspected with a short source/test link, or leave it open. Promote a genuinely new capability or defect to the **review register** only after evaluating overlap and priority. Do not duplicate an existing item or silently turn an observation into a release requirement.

## Topics to revisit

### 1. Forward functionality and supported input combinations

- [ ] Map 2D/3D `compute`, `locate`, `ghost_cells` across ordinary/power weights, available domain types, partial/full periodicity and optional output switches. Note unsupported combinations rather than treating every Cartesian-product cell as promised.
- [ ] Recheck site IDs, duplicate/hidden generators, outside-domain inputs, ghost/persistent weight conventions, and basic failure/empty-result behavior against tests. Cross-reference R-Q01, R-G01/G02 and baseline R-B01–B03.

### 2. Geometry, periodic identities and topology

- [ ] Separate cell-local vertices/edges/faces, image-qualified neighboring contacts and a global periodic complex; identify which representations are available versus inferred. Cross-reference R-O01, P-01a and R-T01–T04.
- [ ] Inspect periodic-image shifts, self-images/winding, clipping/domain boundaries, missing or tiny facets, degeneracies and empty/zero-measure cells. Distinguish native consistency from independent geometric completeness.

### 3. Results, ownership, annotations and normalization

- [ ] Compare raw cell dictionaries, normalized `TessellationResult`, 2D/3D output flags and optional metadata. Check absent versus unknown fields and whether requested geometry is sufficient for post-processing.
- [ ] Inspect mutable ownership/aliasing, source versus primary coordinates, face/edge ordering and neighbor-shift conventions. Cross-reference R-M01, R-A01 and P-02.

### 4. Inverse fitting and realized support

- [ ] Exercise combinations of row observation spaces, penalties/losses, bounds, fixed/equality data, per-row applicability/shape, zero strengths, gauges and disconnected components (R-I01–R-I03). Preserve original row identity when subsets change.
- [ ] Distinguish optimization residuals, mandatory/tentative support, realized image-qualified faces, connector visibility, active-set updates, failure reasons and final returned-state validity (R-I04–R-I06). Record unsupported combinations rather than promising a new constrained solver.

### 5. Derived geometry and visualization-facing data

- [ ] Inventory existing 2D/3D cell and face properties, inside/outside tests, nonnegative point-to-**finite-segment** distance and signed connector-clearance semantics. Include degenerate geometry and numerical-tolerance cases.
- [ ] Assess candidate per-edge distance profiles, face shape/width, centroids/moments, solid angles and the information needed by visualization/mesh exports (R-O05a, P-04a/P-04b). Keep ChemVoro-specific artifact interpretation downstream.

### 6. Numerical boundaries, usability and missing scenarios

- [ ] Revisit scale extremes, anisotropic domains/remapping, near-degeneracies, native Voro++ failure boundaries, overflow and diagnostic availability. Use reproducible examples and distinguish upstream limitations from pyvoro2 policies (R-L03, R-O02/R-O03, R-I06).
- [ ] Check for redundant public knobs, missing examples or user workflows, performance cliffs and 2D/3D asymmetries; record potential enhancements without inferring that all need implementation (R-A01, R-O05b, P-03).

## Recording an outcome

For each inspected question, capture only: **coverage/limitation**, **source or test evidence**, **impact**, and **disposition** (`covered`, `follow-up needed`, or `candidate for register`). Any subsequent accepted scope belongs in the [review register](review-register.md) and, where necessary, an ADR or focused implementation task. This checklist itself never authorizes an API change.
