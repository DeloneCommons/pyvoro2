# 0024 — External native artifact qualification

- **Status:** Accepted contract; issue #88 implementation acceptance pending
- **Date:** 2026-09-28
- **Related issue:** [#88 — native qualification hardening](https://github.com/DeloneCommons/pyvoro2/issues/88)
- **Related plan:** [Phase B qualification hardening](../plans/v0.9.md#phase-b-qualification-hardening-generalize-native-certificate-profiles)
- **Implementation and evidence:** [native qualification workflow](../native-qualification.md)
- **Related decisions:** [ADR 0021](0021-wp5-native-occurrence-and-exact-face-certification.md),
  [ADR 0022](0022-wp6-source-certified-planar-edge-provenance.md), and
  [ADR 0023](0023-wp7-certified-ghost-boundaries.md)

## Context

The accepted WP5–WP8 proofs require particular source operations and execution
properties. A compiler-version allowlist, requested CMake flags, a native
metadata dictionary, or matching final geometry cannot establish those
properties for an installed module. Late flags, source pragmas, response files,
compiler launchers, link processing and wheel repair can change the relevant
execution or payload. Import-time FP checks also cannot establish the current
executing thread's state after import or after a foreign conversion callback.

Issue #88 fixes these qualification boundaries. It preserves the accepted
mathematical objects, source occurrence association, noninterference obligations,
public APIs and structured failure families. It does not accept Checkpoint B,
enter Phase C, or authorize functional vendored-source changes.

## Decision

### Separate approval, measurement, issuance and admission

Qualification uses externally attested artifacts (Model P), with route-specific
components. The controlled build and finalization workflow issues a detached
record; the native module does not certify itself.

| Responsibility | Authority |
|---|---|
| Source approval | Separately reviewed source, schema, consumer and policy identities. Automatic measurement cannot approve changed source. |
| Source measurement | A conservative manifest covers vendor sources, binding/observer code, headers, build configuration, runtime consumers and qualification machinery. Unknown additions in the covered trees enter the measurement. |
| Effective build | Reviewed platform adapters record the actual compiler/backend, assembler and linker execution, dependencies and outputs. |
| Issuance | A controlled external finalizer verifies approved source, complete build evidence, component evidence and final installed payload before writing a record. |
| Admission | The private verifier binds that record to the trusted installation, current loaded native module, ABI, consumers and required components. |
| Runtime environment | A binding-owned raw guard checks the current executing thread before potentially unsafe numeric work and after foreign callback boundaries. |

The source approval document, generated installation anchor and detached record
are not inputs to their own recursively defined source digest. The record binds
the approval identity separately, together with the policy revision, component
schemas and consumer identity. Native embedded identities are consistency
checks against these independently established authorities.

### Effective commands are evidence, requested options are intent

Capture ordered effective argv, expanded response-file contents, relevant
environment, driver/backend/linker identities, actual include dependencies,
source-local option effects, object identities and link inputs/outputs. Reject
opaque wrappers, missing child commands, unobserved response-file changes,
unreviewed options and incomplete evidence. An earlier unsafe option may be
overridden only when its final effective state is established. No favorable
compiler label or CMake cache value compensates for missing evidence.

Each primary translation unit must belong to the independently approved source
closure. External C/C++ headers require an independently established compiler,
SDK, Python or pybind11 provider root plus the actual consumed file identities.
Capturing a header hash records what was read; it does not approve that header.
A candidate `-I` or `-include` option cannot create a trusted provider. Likewise,
every actual linker object, including objects appearing only in expanded
response files, needs an observed translation unit or reviewed toolchain CRT
provenance. Unknown objects and archives refuse qualification.

Source-local optimizer, target and FP controls use a closed reviewed grammar;
unsupported spellings refuse, including controls expanded through preprocessing.
Candidate `-D`/`-U` options cannot replace compiler-owned format, evaluation or
fast-math premise macros with favorable values. The build evidence must establish
those premises from the actual reviewed toolchain and effective execution.

The reviewed source-operation contract excludes proof-sensitive contraction,
reassociation, finite-only assumptions, signed-zero changes, incompatible excess
evaluation and LTO/IPO across the boundary. Independent exact arithmetic
discriminators and relevant optimized instruction inspection supplement the
command and source evidence. Same-build stock/observer noninterference remains
required; whole topology equality across different conforming compilers is not
a general requirement.

Compiler family, exact version, target and SDK remain reproducibility provenance.
They select a reviewed property adapter rather than a compiler patch-version
allowlist. An adapter's existence alone does not qualify an artifact. Existing
AppleClang/macOS and MSVC/Windows WP5/WP8 support must be preserved with actual
positive evidence. WP6/WP7 may explicitly refuse where their separate occurrence
and noninterference evidence is absent.

### Trust the final installation identity

The finalizer writes a detached canonical record and a generated Python
installation anchor containing its digest and installation identity. The anchor
is delivered through the trusted package finalization/installation path.
Placing arbitrary JSON beside a module does not grant qualification. The
record names exact native module and relevant bundled dependency hashes,
approved consumers, target/ABI, build-evidence identity and component-evidence
identities. A different conforming rebuild may receive a new valid record;
historical native bytes are not a permanent allowlist.

Register a native module at its import boundary and require an immutable
installation for its loaded lifetime. Admission checks the actual imported
module, its registered file identity and the bound dependencies and consumers.
Linux additionally checks mapped device/inode identity. Replacing files after
registration invalidates admission. Immutable identity checks may be cached;
the executing thread's mutable FP environment may not.

This is a numerical/build trust boundary, not a native-code security sandbox.
Replacing the trusted verifier and installation anchor together, a malicious
compiler, or arbitrary code injection is outside this model. Qualification adds
no mandatory runtime dependency, public override or unsafe bypass.
System toolchain/runtime providers and OS loader/search-cache provenance remain
bounded controlled-platform assumptions recorded by the adapter. Appearance in
an include path or loader cache alone does not approve an arbitrary provider.
This does not extend qualification into attestation of the entire operating
system.

### Raw runtime inspection precedes coercion

On GNU x86-64, admission requires the following independently inspected raw
controls. Masks use integer bit operations; no arithmetic probe precedes them.

| Control | Required predicate |
|---|---|
| x87 rounding | `(CW & 0x0c00) == 0` |
| SSE rounding | `(MXCSR & 0x6000) == 0` |
| SSE FTZ/DAZ | `(MXCSR & 0x8040) == 0` |
| x87 exception masks | `(CW & 0x003f) == 0x003f` |
| SSE exception masks | `(MXCSR & 0x1f80) == 0x1f80` |
| Declared 64-bit-precision extended `long double` | `(CW & 0x0300) == 0x0300` |

Apple and MSVC adapters inspect their reviewed active evaluation domains and
controls. Runtime adapters and their optimized dispatch/refusal paths require
platform-specific evidence. Sticky exception status is ignored for admission
and preserved; the guard never normalizes caller state or masks traps.

Check public entry before package-owned floating preparation, native entry
on raw `args`/`kwargs` before arity, keyword validation, error formatting or typed
pybind conversion, and replay/enclosure continuation before its floating
operations. Malformed arguments must not cause caller representations to run
before the raw guard. A small private `_fpguard` extension is loaded at package
initialization and retained for the first public forward check, before lazy
loading `_core` or `_core2d`. Their geometry imports remain lazy for pure Python
and inverse use. An absent guard allows those imports but explicitly refuses
forward execution. The guard is a third native payload bound by the artifact
record, not an external runtime dependency. Initial interpreter/Python/NumPy
module loading under an already hostile FP environment is outside the protected
forward-operation boundary; those initialization steps must finish before
callers install such a state.

Materialize foreign array, scalar, indexing and sequence
protocol results under guarded boundaries. Recheck immediately after each
foreign callback, including exception returns, before another conversion or
precondition calculation can execute. Checking only after a complete NumPy
conversion is too late when a callback returns signaling-NaN data under an
unmasked trap. Raw profile inspection must itself remain safe in hostile state.
Warning delivery is also a foreign callback boundary: recheck after a
caller-installed warning handler returns or raises before numeric work resumes.
Within protected native-facing work, domain subclass attribute access,
properties and overridable methods are foreign boundaries too. Check
attribute/method acquisition separately from invocation, and canonicalize
returned numerical containers before arithmetic. Guard overridable reads inside
base domain methods before native-facing calculations resume. Apply the existing
runtime guard when preparation, certificate packaging, diagnostics, normalization
and wall validation consume domain values. Legitimate overrides remain active.
Standalone domain/algebra use retains its existing contract and lazy geometry
imports, without acquiring certificate components.

The check is per call in the executing thread, including worker threads and
calls after FP state changes. No import-time success or another thread's result
can substitute for it.

### Compose only the consumed route components

| Certificate consumer | Required components |
|---|---|
| Ordinary spatial source attribution/audit | `wp5-spatial` |
| Ordinary planar occurrence provenance | `wp6-planar` |
| Spatial selected ghost boundaries | `wp5-spatial` + `wp7-spatial` |
| Planar selected ghost boundaries | `wp6-planar` + `wp7-planar` |
| Spatial locate owner-image enclosure | `wp8-spatial` |
| Planar locate owner-image enclosure | `wp8-planar` |

WP6 remains independently owned; it does not inherit WP5's reverse-plane
argument. WP7's selected route does not inherit qualification merely because
its ordinary producer passed. WP8 retains separate source/enclosure
qualification. Its bounded final NumPy/BLAS transform continues to allow the
association/FMA behavior already included in the enclosure proof; strict native
source ordering must not erase that allowance.

Safe geometry-only ghost and ID-only locate routes retain their own safety and
insertion obligations without acquiring unused certificate components.
Zero-query routes construct no native geometry and issue no native certificate.
Ordinary planar qualification refusal cannot become successful empty geometry
or an uncertified owner-bearing fallback.

Missing or untrusted qualification, source/schema mismatch, effective-build
failure, target/ABI mismatch, changed payload, missing components and current
runtime FP mismatch remain distinct private reasons. Owning routes translate
them into the existing WP5, WP6, GHOST and LOCATE families. Hard refusal is
atomic: no certified prefix, guessed provenance, or silently repaired result.

### Requalify changed inputs and final distributions

Changes to the approved closure, schema/consumer interpretation, execution
properties, toolchain adapter, native payload or bundled dependency invalidate
the affected qualification. Shared closures require affected ordinary and
selected-route noninterference checks. Recomputing a hash is not requalification.
Repair, stripping, bundling and signing belong to explicit postprocessing
lineage. Records and installed smoke tests must refer to the resulting native
bytes, not a pre-repair donor.

Keep source approval, optimized native qualification, Python compatibility,
positive route evidence, negative controls, hostile-runtime tests, sanitizers
and distribution checks distinguishable. Expensive native evidence may be
reused across Python minors only with an explicit proof-relevant equivalence
argument plus per-artifact ABI/schema/import checks. Normal CPython 3.10–3.14
coverage remains required. Sanitizer success does not establish optimized
arithmetic or noninterference.

The controlled qualification runner owns assertion execution, test selection
and plugin loading. Inherited `PYTHONOPTIMIZE`, `PYTEST_ADDOPTS`, selection/filter
or plugin controls cannot disable assertions, skip required tests or reduce the
suite that the finalizer accepts. Record the actual command and completed
component coverage; an externally supplied success report remains insufficient.

Issue #88 acceptance requires the actual final repaired/installed GNU 14.2.1
manylinux positive, GNU 13.3 historical control, strict AVX/FMA-capable positive,
unsafe discriminators, and preserved AppleClang/MSVC WP5/WP8 positives. Retain
the original archived corpus rather than regenerating expected provenance:
WP6 current 48 cases/1718 occurrences, archived WP6 92/2298, and selected planar
WP7 30/287, alongside spatial and WP8 regressions. Missing acceptance-critical
platform or artifact evidence is a blocker, not successful qualification.

## Consequences and alternatives

Certificate availability becomes a property of a reviewed, evidenced installed
artifact and current execution, without changing the accepted geometry. Custom
or ordinary unqualified builds can refuse certificate consumers even when their
metadata resembles a qualified build. Rebuilds need controlled evidence and
finalization; users receive no compiler selector or correctness override.

Retaining exact compiler-version literals, approving automatic hash refreshes,
trusting requested flags or native self-metadata, accepting an adjacent unsigned
claim as authority, normalizing caller controls, and treating a refusal-only
wheel smoke as positive qualification are rejected. Each omits an independent
obligation above. This ADR fixes the accepted contract; its status does not
claim that issue #88's final evidence or independent review has completed.
