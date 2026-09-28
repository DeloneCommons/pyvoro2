# Native qualification workflow

[ADR 0024](decisions/0024-external-native-artifact-qualification.md) owns the
qualification policy. This page maps the implementation and evidence needed to
review a build. Issue [#88](https://github.com/DeloneCommons/pyvoro2/issues/88)
acceptance remains pending; this page does not qualify a platform or artifact.

## Implementation map

| Component | Responsibility |
|---|---|
| `tools/native/qualification/source_policy.py` | Measure the conservative source/consumer closure and compare separately reviewed approval. |
| `src/pyvoro2/_internal/native_approval.json` | Reviewed source, schema, consumer and component identities; measurement never updates approval. |
| `tools/native/qualification/record_command.py`, `adapters.py`, `effective_build.py` | Record actual tool execution and verify effective operations, dependencies, objects and link lineage. |
| `tools/native/qualification/finalize.py` | Verify controlled evidence, execute the repository route suite, bind final installed payload and issue the detached record and installation anchor. |
| `src/pyvoro2/_internal/native_qualification.py` | Verify trusted record, source/schema/consumer compatibility, ABI, immutable loaded payload and route composition. |
| Private `_fpguard`, `cpp/native_runtime.hpp`, `src/pyvoro2/_internal/native_runtime.py` | Raw current-thread control inspection before lazy geometry import, guarded foreign coercion and route-owned refusal. |
| `src/pyvoro2/_internal/domain_access.py` and domain consumers | Guard subclass attribute/method acquisition and invocation separately before numerical continuation. |

The source checkout's default installation anchor is unqualified. A normal
editable build, favorable `_qualification_identity()` metadata, or copying a
JSON record beside an extension cannot issue qualification. Finalization is an
external controlled build step, not a runtime fallback. It adds no runtime
dependency. Installation immutability and the trusted-anchor boundary are
specified in ADR 0024; this mechanism is not a security sandbox against arbitrary
replacement of the package and its verifier.

Package initialization retains `_fpguard` before forward operations can load
the lazy geometry extensions. The issued record binds all three native
payloads. Pure Python/inverse imports remain available without geometry
extensions; forward operations explicitly refuse if the raw guard is missing.
Initial interpreter and Python/NumPy loading under an already hostile FP state
is outside the protected operation boundary. Runtime tests import the package
before installing hostile controls, then exercise the first forward call as
well as subsequent calls. No path normalizes caller controls.

## Source approval and fresh builds

Measure before review and check the separate approval before qualification:

```bash
python tools/native/qualification/source_policy.py --root . --measure
python tools/native/qualification/source_policy.py --root . --check-approval
```

The first command reports identities only. The second must fail while approval
is absent or stale. Review the entire changed closure and its proof impact
before updating approval; do not automate approval from the newly measured
hash. A source, consumer, schema or shared binding change can invalidate several
route components. Keep unchanged historical evidence intact.

Rebuild native modules after every relevant C++/CMake change and record their
actual import paths. Tests against a previously installed extension do not
validate changed native source. Source-policy measurement and runtime native
identity are complementary checks; neither replaces effective-build evidence.

## Effective build and finalization

The compiler launcher interface is:

```text
python tools/native/qualification/record_command.py --output-dir RECORDS -- COMPILER ARGV...
```

The controlled workflow must collect every relevant compile and link invocation.
Preserve ordered argv, environment, response files, actual compiler/backend and
linker identities, include dependencies, source-local options, preprocessing
evidence, objects and native outputs. GNU child observation, controlled Clang
jobs, and the MSVC process adapter must establish the actual executed work.
An unsupported wrapper or incomplete record refuses qualification. Requested
`NativeFP.cmake` flags and verbose logs alone are insufficient.

Input identity and approval have different roles:

| Actual build input | Required authority |
|---|---|
| Primary C/C++ translation unit | Membership in the independently approved source closure. |
| External header | Compiler/SDK, Python or pybind11 provider root established independently by the controlled adapter, plus the consumed file's identity. Candidate include flags cannot establish that provider. |
| Linker object, including a response-only input | Observed translation-unit output or reviewed toolchain CRT provenance. |
| Archive | Reviewed provenance; unknown archives refuse qualification. |

A captured hash alone does not approve arbitrary external source. Keep
source-local optimizer/target/FP controls within the reviewed closed grammar,
including their preprocessed forms, and reject candidate definitions or
undefinitions that spoof compiler-owned proof-premise macros. Record relevant
system-provider and OS loader/search-cache provenance within the bounded trusted
platform; the workflow does not attest every operating-system file.

After wheel repair and clean installation, finalization consumes the observed
build records and the explicit postprocessing receipt:

```text
python tools/native/qualification/finalize.py \
  --source-root SOURCE \
  --installation-root SITE_PACKAGES \
  --records-dir RECORDS \
  --postprocess-receipt RECEIPT_JSON \
  --corpus ORIGINAL_CORPUS \
  --output FRESH_EVIDENCE_DIRECTORY
```

These are workflow interfaces, not evidence that the command has passed on this
branch. The finalizer runs the repository-owned route runner itself and refuses
caller-submitted success assertions. The runner owns its assertion, selection
and plugin policy: inherited `PYTHONOPTIMIZE`, `PYTEST_ADDOPTS`, test filters and
plugin controls must not reduce required coverage or disable assertions.
Retain actual commands and completed component coverage in the evidence.
Qualification-only candidates belong to
isolated evidence processes; production modules must contain no candidate hook
or bypass. Run an unmodified installed positive smoke after record issuance.
The final wheel must contain exactly the qualified native payload and trusted
record/anchor, with wheel integrity metadata regenerated by the packaging step.

Record every repair, strip, bundling or signing step and its input/output hashes.
Bind relevant bundled native dependencies as well as the imported extensions.
Direct wheel and sdist-to-wheel installations each require artifact-specific
identity/ABI/route checks. Reusing native evidence between Python minors requires
an explicit equivalence argument, not matching version labels.

`check_dist.py --require-qualification` and
`check_wheel_matrix.py --require-qualification` require the detached record in
each production wheel. These packaging checks do not issue or authenticate a
qualification claim; the finalizer and installed verifier retain that authority.

Native dispatch receives raw `args`/`kwargs` so FP inspection precedes arity,
keyword and error-formatting paths as well as typed conversion. Runtime tests
also cover state changes in warning handlers: warning delivery must recheck the
thread before subsequent numeric work, including when the handler raises.
In protected native-facing workflows, domain-state regressions exercise subclass
getters/properties, method lookup and method return. Existing runtime guards also
cover overridable reads inside base domain methods, with returned numerical
containers canonicalized before arithmetic in preparation, certificate
packaging, diagnostics, normalization and wall validation. Supported overrides
still execute. Standalone domain/algebra use retains its existing FP contract
and needs no certificate or `_core`/`_core2d` import. Independent runtime review
remains active.

## Required evidence and support boundary

| Evidence | Issue #88 requirement |
|---|---|
| GNU control | GNU 13.3 strict positive. |
| GNU manylinux | Actual GNU 14.2.1 final repaired, installed production wheel with positive certificate routes. An extracted compiler or pre-repair donor is insufficient. |
| Arithmetic | Strict AVX/FMA-capable positive plus unsafe contraction and power-order discriminators with independent expected bits. |
| Existing platforms | AppleClang/macOS and MSVC/Windows WP5/WP8 positive artifacts through reviewed adapters. These positives remain pending until their actual jobs pass. |
| Planar occurrence evidence | Current WP6 48/1718, original archived WP6 92/2298 and selected planar WP7 30/287 cases/occurrences. |
| Other routes | Accepted ordinary/selected spatial noninterference and distinct WP8 source/enclosure regressions. |
| Runtime | Split x87/SSE rounding, all rounding directions, FTZ/DAZ, PC24/PC53/PC64, trap masks, masked sticky flags, worker threads, after-import changes and callback boundaries. |
| Package | CPython 3.10–3.14 compatibility/full suites, sanitizers, direct/sdist wheels, actual import identity and absent production bypass hooks. |

`build.py --sanitizers` emits separate route, safety and distribution reports
scoped `sanitizer-safety-only`. It retains the byte-identical unqualified wheel,
issues no production qualification record or trusted installation anchor, and
checks refusal in fresh installed processes before and after the run. Production
modules and the strict arithmetic control retain instrumentation, verified from
actual compile/backend records; the fixed safety suite and route/corpus checks
remain required. Both deliberately unsafe GNU companion translation units
explicitly record `-fno-sanitize=all`, preserving the frozen power-order
expectations (strict `0/1`, unsafe `1/1`) when instrumentation changes unsafe
reassociation. Their inherited production vendor objects and sanitizer link
flags remain unchanged. This safety path makes no raw-guard proof claim.
Optimized release qualification still requires the unchanged strict guard
inspection; sanitizer safety does not establish optimized release evidence.

Adapter availability is not a support claim. WP6/WP7 on a family without the
required occurrence evidence explicitly refuse. Existing WP5/WP8 Apple/MSVC
support and final repaired GNU14 qualification are acceptance gates, not optional
future portability. Linux aarch64 and other new targets remain outside #88.
Raw runtime tests with dangerous unmasked traps run in disposable processes;
surviving harnesses restore exact saved state. The guard never changes it.

## Evidence navigation and handoff

- [Issue #88 implementation map](issue88-implementation.md) records the verified
  baseline, predecessor archive identity and implementation checklist.
- [WP5](wp5-implementation.md), [native witness](native-face-witness.md),
  [WP7](wp7-implementation.md) and [WP8](wp8-implementation.md) retain their
  accepted route-specific source and mathematical evidence.
- The final review handoff must identify PR number, exact head/root tree and
  expected `dev` base, source/approval/schema manifests, commands, toolchains,
  effective records, component results, native/dependency/wheel hashes, imported
  paths, CI run IDs, reproduction commands and complete evidence checksums.

Keep historical predecessor characterization separate from tests actually rerun
on the final implementation head. Do not sum overlapping test selections or
claim a changed repaired payload from its donor's evidence. Until every required
gate and independent review is recorded, #88 and Checkpoint B remain unaccepted.
