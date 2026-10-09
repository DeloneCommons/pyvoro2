# Potential work

This file records candidate improvements discovered or reconsidered during the maintainer reassessment.

**Status:** Non-normative working notes. Nothing here is an approved implementation requirement or release commitment.

## Global topology and periodic image reconstruction

**Status:** Research candidate.

**Motivation**

Voro++ computes cells individually. pyvoro2 already provides numerical vertex/topology normalization and periodic neighbor-image reconstruction, but these operations do not establish a universally exact, complete global complex.

Investigate how to construct a globally consistent vertex/edge/face representation and assign correct lattice shifts to periodic neighbor relations.

**Candidate direction**

- Re-examine existing vertex/edge globalization algorithms and their identity assumptions.
- Investigate reconstructing global incidence from cell-local vertices, edges, and faces.
- Investigate recovering periodic translation vectors through corresponding boundary relations and consistent lattice-image assignments.
- Distinguish geometric coincidence, numerical matching, exact identity, and complete semantic coverage.
- Preserve periodic winding and self-neighbor relations.
- Validate candidate algorithms using independent mathematical oracles, including known degeneracies and missing-positive-face cases.

Important: matching native faces alone cannot recover an exact positive face omitted by the backend. Reconstruction from native incidence and direct construction from power/Laguerre halfspaces are distinct approaches to evaluate.

**Initial code entry points**

- `src/pyvoro2/normalize.py`
- `src/pyvoro2/_internal/normalization.py`
- `src/pyvoro2/_internal/spatial/face_shifts.py`
- `src/pyvoro2/_internal/planar/edge_shifts.py`
- `cpp/bindings.cpp`
- `cpp/bindings2d.cpp`

**Historical evidence**

See the reboot requirements inventory (R-T01–R-T04) and evidence index (E06–E10). Historical implementations are evidence, not mandatory architecture.

**Open question**

What useful levels of global identity, completeness, and periodic-image correctness can pyvoro2 guarantee at acceptable computational cost?

No specific algorithm, public API, or certification policy has yet been selected.
