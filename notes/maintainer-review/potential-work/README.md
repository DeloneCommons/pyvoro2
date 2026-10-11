# Potential work

Only substantial new or materially expanded scientific/engineering investigations belong in this directory. The [decision register](../review-register.md) is the **single status index**; neither a candidate nor its historical implementation authorizes new code, API changes, or release commitments.

| Note | Distinct purpose |
| --- | --- |
| [`geometry-layers.md`](geometry-layers.md) | Independent local/adjacency/global geometry layers and ChemVoro density/surface integration studies. |
| [`large-weight-numerics.md`](large-weight-numerics.md) | Concrete native large-radius failure and potential Voro++ upstream investigation; tracked as R-O03. |
| [`geometry-output-presets.md`](geometry-output-presets.md) | Optional output ergonomics separate from historic metadata/certification work. |
| [`native-performance.md`](native-performance.md) | Read-only contexts, 2D ghost batches and grid-locate hypotheses, distinct from old Fraction optimization. |

The former `global-topology.md` repeated historical [R-O01](../../../docs/development/reboot/requirements-inventory.md#r-o01) in most respects; its ChemVoro motivation is now summarized in [the register](../review-register.md) and expanded independently in `geometry-layers.md`. The former `development-qualification-boundaries.md` largely restated accepted [ADR 0027](../../../docs/development/decisions/0027-v0.9-reboot-from-v0.8.md) and R-E03/PA-003; its extra local-test boundary question is preserved in the register. Both removed notes remain recoverable from Git history.

Historical implementations and accepted *historical* checkpoints remain evidence, not reboot requirements: [historical knowledge guide](../../../docs/development/reboot/index.md).
