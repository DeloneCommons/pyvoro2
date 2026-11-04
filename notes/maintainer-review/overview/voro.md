# Voro++: core model

**Baseline:** the vendored Voro++ in pyvoro2's v0.8-based reboot. Its 3D code follows the Voro++ 0.4.6 family (with a downstream-included upstream pruning correction); its 2D implementation is separate.

## How a cell is computed

- **Cell-by-cell:** the diagram is constructed as independent cells, not as a shared global mesh.
- **Cell representation:** vertices are shared *within each cell* and connected through edges; faces are recovered by traversing this connectivity. The 2D structure is simpler than the 3D polyhedron structure.
- **Initial geometry:** a bounded convex region around the current generator (a point/site); domain boundaries or periodic self-images constrain it.
- **Clipping:** each relevant other generator contributes a separating half-space. A clipping operation removes the opposite side and updates vertices, edges, and their links.
- **Finding candidates:** the container stores generators in spatial blocks. `voro_compute` scans selected blocks and uses radius/distance bounds plus shape-aware region tests to avoid irrelevant cuts. This is not a global Delaunay-first construction.
- **Power/Laguerre:** the same clipping mechanism uses the cut parameter `|p_j-p_i|² + w_i - w_j`, with backend weights represented as squared radii `w_i = r_i²`. Cells may be empty or exclude their own generator.
- **Periodicity:** periodic images (including self-images) are considered. Rectangular mixed-periodic boxes and fully periodic general 3D lattices have different container implementations.
- **Output:** native geometry is per-cell (volume/area, vertices, faces/edges, neighbor site IDs). A globally identified vertex/edge/face complex is not built automatically; ordinary neighbor IDs do not specify lattice-image shifts.

**Numerical boundary:** binary64 arithmetic and geometric tolerances can affect both which cuts are made and the resulting topology. See [contracts](../contracts/voro.md).

## Source entry points

| Role | Source |
| --- | --- |
| Cell clipping and topology | [3D `cell.cc`](../../../vendor/voro++/src/cell.cc) · [2D `cell_2d.cc`](../../../vendor/voro++/2d/src/cell_2d.cc) |
| Candidate search and pruning | [`v_compute.cc`](../../../vendor/voro++/src/v_compute.cc) |
| Weighted calculation policy | [`rad_option.hh`](../../../vendor/voro++/src/rad_option.hh) |
| Domains and periodicity | [`container.hh`](../../../vendor/voro++/src/container.hh) · [`container_prd.cc`](../../../vendor/voro++/src/container_prd.cc) · [`unitcell.cc`](../../../vendor/voro++/src/unitcell.cc) |
| pyvoro2 native boundary | [`cpp/bindings.cpp`](../../../cpp/bindings.cpp) · [`cpp/bindings2d.cpp`](../../../cpp/bindings2d.cpp) |
