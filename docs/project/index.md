<div class="pv-overview" markdown="1">

# About pyvoro2

<div class="pv-overview-lead" markdown="1">

Open scientific Python software for studying space through **Voronoi and
power/Laguerre tessellations**: from 2D and 3D periodic geometry to fitting
power weights from pairwise separator observations.

</div>

pyvoro2 supplies geometry and inspectable numerical results for research
applications. Domain-specific models and interpretation belong in downstream
software, such as chemvoro, rather than in the geometric core.

## Scientific foundations and software

**Voro++ computes the cells.** pyvoro2 builds on vendored 3D and legacy 2D
Voro++ sources through separate native extensions. The 3D snapshot includes
the upstream radical-pruning robustness fix for periodic power diagrams,
recorded in the [changelog](changelog.md).

**pyvoro2 connects geometry to scientific workflows.** Its Python layer adds
domain handling, validation, normalized topology, image-labelled periodic
contacts, and separator-based inverse fitting with structured diagnostics.
Forward computation is independently useful; inverse fitting estimates
weights at fixed sites. A good separator fit does not by itself establish
that the requested boundaries exist in the resulting diagram.

The 3D and planar interfaces have distinct domain and output capabilities.
Forward operations are stateless: each builds a backend container and returns
Python data. See [capabilities and limitations](../guide/capabilities.md),
[operations](../guide/operations.md), and the
[mathematical foundations](../theory/index.md) for these contracts and their
scientific interpretation.

## Project status

<div class="pv-overview-grid" markdown="1">
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">Published · v0.8.0</p>

The public release is the baseline for current behavior. The package remains
beta; API lifecycle status varies by surface.

[Release notes](release-notes-v0.8.md) ·
[API lifecycle](../development/api-lifecycle.md)

</div>
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">In development · v0.9</p>

Development restarts from v0.8.0 under
[ADR 0027](../development/decisions/0027-v0.9-reboot-from-v0.8.md).
Future geometry architecture and release scope remain undecided.

[Roadmap](roadmap.md) · [Changelog](changelog.md)

</div>
</div>

## Project information

<div class="pv-overview-links" markdown="1">
<div markdown="1">

**License and attribution**

pyvoro2-authored code is LGPLv3+ from v0.6.0; earlier releases remain MIT.
Vendored Voro++ retains its separate upstream license.

[License and notices](license.md)

</div>
<div markdown="1">

**Contributing**

Development guidance covers tests, notebook maintenance, architecture, and
the review process.

[Contributor setup](https://github.com/DeloneCommons/pyvoro2/blob/dev/CONTRIBUTING.md) ·
[Development](../development/index.md)

</div>
<div markdown="1">

**Policy and provenance**

AI assistance is disclosed; the maintainer reviews integrated changes and
remains responsible for the software and scientific claims.

[AI-assisted development](ai.md) ·
[Decision records](../development/decisions/index.md)

</div>
</div>
</div>
