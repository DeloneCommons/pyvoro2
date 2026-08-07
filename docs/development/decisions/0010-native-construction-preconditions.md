# 0010 — Strict native construction preconditions

- **Status:** Accepted
- **Date:** 2026-08-05
- **Related issue:** [#38 — v0.8 R3-A: add strict shared validators and guard every native container construction](https://github.com/DeloneCommons/pyvoro2/issues/38)
- **Related decisions:** [ADR 0005](0005-tessellation-result-contract.md), [ADR 0006](0006-v0.8-cleanup-release.md)
- **Related plan:** [v0.8 remediation execution plan](../plans/v0.8-remediation.md)

## Context

Every spatial or planar `compute`, `locate`, and `ghost_cells` call constructs
a fresh Voro++ container. The native constructors historically trusted values
that had already crossed pybind11. In particular, zero or negative
`init_mem`, overflowing block products, non-finite arrays, and unsafe
constructor arithmetic could reach allocation or backend computation. A
direct call to the internal `_core` or `_core2d` module could also bypass
Python-wrapper checks.

This is a process-safety boundary, not a new tessellation feature. R3-A of
issue #38 must make every one of the 12 spatial and 6 planar construction
routes reject unsafe converted inputs before constructing Voro++. R3-B remains
responsible for broad adoption of exact-type validation and immutable
ownership across IDs, shifts, options, masks, tolerances, inverse models, and
all domain values.

## Decision

### Exact controls and validation layers

At public Python entry points, native integer controls use `operator.index`
semantics. Python and NumPy integer scalars and other genuine index-protocol
scalars are accepted. Booleans, floating-point values, strings, complex
values, and scalar arrays are rejected rather than converted lossily.
`init_mem` and every explicit block count must be positive and no greater than
the C++ `int` maximum. `block_size`, when supplied, must be a positive finite
real scalar; it is not an integer field.

Python owns exact source-type policy because original Python type information
may be lost during pybind11 conversion. The C++ boundary owns safety of the
values and arrays after conversion: positive controls, C++ destination ranges,
array shapes and lengths, internal ID invariants, finite coordinates and
queries, non-negative finite radii, domain parameters, checked constructor
arithmetic, and bounded known eager allocation. The internal native modules
are not public APIs, but every direct-native construction route receives this
defense in depth.

Validation occurs before the operation that needs the value. Public wrappers
validate the operation mode and native controls before array coercion, then
validate finite point/query/radius data and domain geometry before block
resolution, numerical work, or native dispatch. Each native route validates
converted controls first, then arrays and destination counts, then geometry
and constructor floating-point arithmetic, then checked integer dimensions
and the resource estimate. Only after all checks succeed may it construct a
Voro++ container. Rejections use `ValueError` with the affected input or
derived quantity in the message.

### Checked native arithmetic

All allocation counts and byte estimates use checked `size_t` addition and
multiplication. Every derived value stored in a Voro++ `int` is separately
checked against the C++ `int` range. This includes:

- `particle_stride * init_mem` (`ps * init_mem` in Voro++);
- rectangular block products, periodic mask dimensions, mask products, and
  queue lengths in two and three dimensions;
- triclinic primary and conservative extended block products, compute-mask
  dimensions, and queue lengths; and
- point and query counts and generated internal-ID ranges.

Finite inputs are not sufficient when a constructor immediately evaluates an
overflowing expression. The preflight therefore also checks finite block
widths and reciprocals, Voro++ worklist squared-distance expressions, the 3D
rectangular maximum squared length, and the periodic unit-cell shell,
tolerance, and squared-norm arithmetic in source order.

### Source-derived resource estimates and cap

The allocation guard estimates the memory that current vendored constructors
and their immediately constructed work arrays are known to allocate eagerly.
For a rectangular container it covers block pointer/counter arrays, the
initial `block_product * init_mem` particle slots (with standard or power
stride), compute masks and queues, the dimension-specific worklist, and wall
pointer storage.

For a 3D triclinic container, the primary block product is
`nx * ny * nz`. The vendored `unitcell.cc` stores doubled vertex
coordinates, forms `q = sqrt(x*x + y*y + z*z)`, and then halves the maxima of
`y + q` and `z + q`. In physical coordinates the constructor extents are
therefore

\[
\max_v(v_y + \lVert v\rVert)
\quad\text{and}\quad
\max_v(v_z + \lVert v\rVert).
\]

A conservative unit-cell covering-radius bound is

\[
R = \tfrac12\left(\lVert a\rVert+\lVert b\rVert+\lVert c\rVert\right).
\]

`R` bounds `||v||`, but it does not directly bound either source extent. Each
source extent is at most `2R`. For Voro++'s lower-triangular basis, the checked
componentwise L1 bound

\[
E = |b_x| + |b_{xy}| + b_y + |b_{xz}| + |b_{yz}| + b_z
\]

satisfies
`||a|| + ||b|| + ||c|| <= E` and therefore bounds both source extents. It
gives `ey = floor(E / by * ny) + 1` and
`ez = floor(E / bz * nz) + 1`, from which the preflight checks the extended
product `nx * (ny + 2*ey) * (nz + 2*ez)`. The non-negative L1 accumulation
uses checked additions in a wide type. Each accumulated bound and the final
bound/period/block scaling are rounded outward toward positive infinity before
`floor` and checked C++ `int` conversion, so round-to-nearest arithmetic cannot
make the intended upper bound smaller.

The estimate covers the extended pointer, counter, and image-flag arrays;
initial particle storage for primary blocks; the periodic compute mask and
queue; the 3D worklist; and the initial unit-cell storage. This bound is
deliberately conservative and traceable to the current `container_prd`,
`unitcell`, `v_base`, `v_compute`, `cell`, worklist, and configuration sources.

The exact cap is `1 << 30`, or 1,073,741,824 bytes, for the aggregate known
eager native allocation estimate. An estimate equal to the cap is admitted;
an estimate greater than the cap raises `ValueError`. There is no unsafe
override. The estimate is not a claim about total peak process memory: it does
not include later dynamic growth, returned geometry, or Python-owned arrays.

### Build and issue boundary

After any C++ binding or preflight edit, the normal fresh native build path is

```bash
python -m pip install -e ".[all]" --no-build-isolation -v
```

An older or donor extension is not compatible evidence for the edited source.
Temporary sanitizer builds must be replaced by this ordinary editable build
before handoff.

This decision completes only R3-A. It does not decide R3-B ownership and
adoption details, periodic nearest-image algorithms (R4), mandatory duplicate
or containment policy (R5), diagnostic severity (R8), or release
qualification (R9 and issue #33). R3 remains open until R3-B and the integrated
R3 gate pass.

## Consequences

- Unsafe controls, malformed converted arrays, overflowing constructor
  arithmetic, and over-cap known eager allocations fail before Voro++
  construction on all 18 native paths.
- Direct internal native calls receive the same converted-value protection as
  high-level calls, while exact original-type semantics remain a Python-layer
  responsibility.
- Large but otherwise well-formed block configurations may be rejected by a
  conservative estimate even when a particular machine has more memory.
- Valid in-cap calls retain the existing Voro++ construction and numerical
  path; no public signature, default, result schema, or tessellation formula
  changes.
- Changes to the vendored constructor allocation layout require review of the
  corresponding estimate and source trace.

## Alternatives considered

### Validate only in Python

Rejected. Internal native modules remain importable, and wrapper regressions
must not turn a direct construction into memory corruption or a backend exit.

### Rely on allocation failure or catch backend errors

Rejected. Integer overflow and zero-length native allocation can occur before
a recoverable allocation exception, and some backend failures terminate the
process.

### Provide an unsafe resource-limit override

Rejected. An override would reopen the exact construction hazard that the
guard is intended to close and would make direct-native safety depend on an
opt-out policy.

### Predict every byte of peak memory exactly

Rejected. Later container growth and output geometry depend on the input
tessellation. A checked, source-derived bound on known eager construction
allocations is auditable and can reject dangerous fixed controls before any
geometry-dependent work.
