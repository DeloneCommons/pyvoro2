# Native allocation-failure exception safety — upstream Voro++

**Status:** `selected for research`, **pre-C++ upstream prerequisite** R-O02b (paired with [R-O03](large-weight-numerics.md)). **No upstream issue, reproducer repo, PR, pyvoro2 native patch or fork has been created or authorized for immediate execution by this note.**

## Source-evident defects to reproduce

[pyvoro2 issue #64](https://github.com/DeloneCommons/pyvoro2/issues/64) records possible allocation leaks when a later `new[]` throws `std::bad_alloc`. The vendored [`container_prd.cc`](../../../vendor/voro++/src/container_prd.cc) matches the inspected upstream `chr1shr/voro` master version. The following are **source-level failure hypotheses, not yet validated fault-injection demonstrations**:

- **Constructor / partial ownership:** `container_periodic_base` allocates several raw arrays in the member initializer list and per-block `id[]`/`p[]` pairs in the body; `container_base` has similar raw arrays and per-block allocations. When construction throws, the most-derived container destructor is not called, so ownership must be managed separately.
- **Growth / empty-block initialization:** `container_periodic_base::add_particle_memory()` and `container_base::add_particle_memory()` allocate new `id[]` followed by `p[]`. Failure of the later allocation can leak the earlier one; the periodic empty-block path also sets `mem[i]` before allocations succeed, potentially leaving an inconsistent live container.

## Workflow before planned C++ edits in pyvoro2

1. **Reproduce independently.** Identify each distinct confirmed error; make **one standalone repository per error**, each with a minimal pure-Voro++ harness, pinned upstream revision, deterministic failure injection at the relevant `new[]` boundary, and clear expected/observed behavior. Test constructor failure versus growth separately and include both periodic and nonperiodic containers; split into separate repositories where the failure or repair materially differs.
2. **Propose an upstream-compatible fix.** Chris Rycroft explicitly prefers **ANSI C++** compatibility in [earlier upstream issue #43](https://github.com/chr1shr/voro/issues/43); avoid C++11-only `unique_ptr` or `nextafter`. Use C++98-compatible local ownership (e.g. careful `try/catch`, cleanup, or a small C++98 RAII guard). Ensure construction failure frees successful prior allocations; growth failure must not leak, corrupt existing arrays, or prematurely commit capacities.
3. **Demonstrate before/after.** In each bug's repository, show original failure and fixed behavior with repeatable commands, fault injection and ASan/LSan where supported. Include normal-success controls and avoid claiming all of Voro++ is globally exception-safe based on these cases.
4. **Then approach upstream.** Only after a focused candidate fix passes its reproducer, prepare/file the corresponding **Voro++ issue for Chris**, with evidence, concise source explanation and the repository link; let the upstream maintainer choose the final patch form. No upstream acceptance or merge is required to finish the **reporting** prerequisite.
5. **Separately decide pyvoro2 integration.** Do not change vendored/native source, wrappers, CI or its documented safety contract automatically. Such work requires a specific maintainer-approved decision and ordinary tests. The broader fork-versus-patch policy remains R-O02a (`pending`).

This **pre-C++ ordering** is recorded in [the decision register](../review-register.md). It also covers [R-O03's independent large-weight reproducers](large-weight-numerics.md); it does not block ordinary Python/docs work. An explicit maintainer decision is required to waive the ordering.
