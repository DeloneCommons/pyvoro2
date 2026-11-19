# Native allocation-failure exception safety — upstream Voro++

**Status:** `selected for research` (R-O02b, alongside [R-O03](large-weight-numerics.md)). The former blanket pre-C++ reporting prerequisite is superseded by [ADR 0028](../../../docs/development/decisions/0028-upstream-voro-vendor-policy.md): compare each concrete defect against pinned upstream `dev` before deciding on a local backport, mitigation or deferral. **No upstream issue, reproducer repo, PR, pyvoro2 native patch or fork is automatically authorized by this note.**

## Source-evident defects to reproduce

[pyvoro2 issue #64](https://github.com/DeloneCommons/pyvoro2/issues/64) records possible allocation leaks when a later `new[]` throws `std::bad_alloc`. The vendored [`container_prd.cc`](../../../vendor/voro++/src/container_prd.cc) matches the inspected upstream `chr1shr/voro` master version. The following are **source-level failure hypotheses, not yet validated fault-injection demonstrations**:

- **Constructor / partial ownership:** `container_periodic_base` allocates several raw arrays in the member initializer list and per-block `id[]`/`p[]` pairs in the body; `container_base` has similar raw arrays and per-block allocations. When construction throws, the most-derived container destructor is not called, so ownership must be managed separately.
- **Growth / empty-block initialization:** `container_periodic_base::add_particle_memory()` and `container_base::add_particle_memory()` allocate new `id[]` followed by `p[]`. Failure of the later allocation can leak the earlier one; the periodic empty-block path also sets `mem[i]` before allocations succeed, potentially leaving an inconsistent live container.

## Per-defect investigation and possible local backport

1. **Reproduce and compare upstream `dev`.** Identify each distinct confirmed error on the current vendored snapshot and a pinned upstream `dev` revision. Prefer a small standalone pure-Voro++ reproducer per independent defect, using deterministic failure injection at relevant `new[]` boundaries, normal-success controls and clear expected/observed behavior. Test constructor versus growth and periodic versus nonperiodic containers where applicable.
2. **Assess a bounded correction where needed.** If upstream `dev` has fixed a relevant case, consider the smallest C++98-compatible backport. If it still reproduces, evaluate severity, supported recovery semantics and a separately approved correction or deferral. Chris Rycroft's [earlier upstream issue #43](https://github.com/chr1shr/voro/issues/43) notes **ANSI C++** compatibility; avoid C++11-only `unique_ptr` or `nextafter`. Correct construction/growth ownership without premature capacity commits.
3. **Demonstrate before/after when patching.** Show original failure and fixed behavior with repeatable commands, independent fault injection and ASan/LSan where supported. Include normal-success controls; do not claim global exception safety from these cases.
4. **Communicate upstream when appropriate.** Share reproducible findings or candidate fixes with Chris when useful and welcome, in the branch/workflow he currently maintains. A new issue against the old upstream `master`, an upstream merge and a separate reproducer repository are **not unconditional gates** for unrelated pyvoro2 C++ work.
5. **Separately decide pyvoro2 integration.** Do not change vendored/native source, wrappers, CI or the documented safety contract automatically. Each local change needs a focused maintainer-approved issue/PR and tests under ADR 0028. The permanent backend-fork and future upstream-`dev` migration policies remain separate decisions.

The [decision register](../review-register.md) now uses **per-defect**
upstream-`dev` triage under ADR 0028 instead of treating this investigation as
a universal gate for every planned native edit. [R-O03](large-weight-numerics.md)
is assessed independently.
