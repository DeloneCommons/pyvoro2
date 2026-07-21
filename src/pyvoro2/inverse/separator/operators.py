"""Public graph and quadratic-operator views for separator problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import ConnectivityDiagnostics, ConstraintGraphDiagnostics, PowerFitBounds


def _scipy_sparse():
    """Import and return ``scipy.sparse`` only for an explicit conversion."""

    try:
        from scipy import sparse
    except ImportError as exc:
        raise ImportError(
            'SciPy is required for sparse separator-operator conversion; '
            'install scipy or use the dense NumPy conversion instead'
        ) from exc
    return sparse


def _vector(value: np.ndarray, n_sites: int, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (n_sites,):
        raise ValueError(f'{name} must have shape (n_sites,)')
    if not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must contain only finite values')
    return vector


@dataclass(frozen=True, slots=True)
class SeparatorObservationGraphView:
    """Oriented multigraph and row data of a separator-fit problem.

    Obtain this provisional view from
    :attr:`SeparatorFitProblem.observation_graph`.  Each observation is one
    distinct edge column, including zero-confidence rows, repeated rows, and
    different periodic images of the same site pair.  The incidence column is
    ``+1`` at ``site_i`` and ``-1`` at ``site_j``.

    Array fields share the read-only arrays owned by the problem or its
    resolved observations, except for the derived read-only
    ``informative_mask``.
    """

    n_sites: int
    site_i: np.ndarray
    site_j: np.ndarray
    observation_indices: np.ndarray
    requested_shifts: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    z_obs: np.ndarray
    rho: np.ndarray
    informative_mask: np.ndarray
    connectivity: ConnectivityDiagnostics

    @property
    def n_observations(self) -> int:
        """Number of observation rows and incidence columns."""

        return int(self.site_i.shape[0])

    @property
    def implied_difference_targets(self) -> np.ndarray:
        """Alias for the implied targets ``z_obs``."""

        return self.z_obs

    @property
    def effective_edge_weights(self) -> np.ndarray:
        """Alias for ``rho = confidence * alpha**2``."""

        return self.rho

    @property
    def informative_graph(self) -> ConstraintGraphDiagnostics:
        """Connectivity induced only by positive-confidence rows."""

        return self.connectivity.effective_graph

    @property
    def informative_components(self) -> tuple[tuple[int, ...], ...]:
        """Informative connected components, including isolated sites."""

        return self.informative_graph.connected_components

    @property
    def isolated_sites(self) -> tuple[int, ...]:
        """Sites isolated in the informative observation graph."""

        return self.informative_graph.isolated_points

    @property
    def relative_component_offsets_identified_by_data(self) -> bool:
        """Whether informative data connect all sites into one component."""

        return bool(self.connectivity.candidate_offsets_identified_by_data)

    @property
    def global_geometric_gauge_identified_by_data(self) -> bool:
        """Always false for separator observations of weight differences."""

        return False

    def incidence_dense(self) -> np.ndarray:
        """Return the oriented incidence matrix as a dense NumPy array.

        The result has shape ``(n_sites, n_observations)``.  Column ``r`` is
        ``+1`` at ``site_i[r]`` and ``-1`` at ``site_j[r]``, so
        ``B.T @ weights == weights[site_i] - weights[site_j]``.
        """

        incidence = np.zeros(
            (int(self.n_sites), self.n_observations),
            dtype=np.float64,
        )
        columns = np.arange(self.n_observations, dtype=np.int64)
        np.add.at(incidence, (self.site_i, columns), 1.0)
        np.add.at(incidence, (self.site_j, columns), -1.0)
        return incidence

    def incidence_sparse(self, *, format: str = 'csr') -> object:
        """Return the incidence matrix in a requested SciPy sparse format.

        SciPy is imported lazily and remains optional.  The sparse storage
        format is a conversion choice, not part of the mathematical contract.
        """

        sparse = _scipy_sparse()
        columns = np.arange(self.n_observations, dtype=np.int64)
        rows = np.concatenate((self.site_i, self.site_j))
        column_indices = np.concatenate((columns, columns))
        data = np.concatenate(
            (
                np.ones(self.n_observations, dtype=np.float64),
                -np.ones(self.n_observations, dtype=np.float64),
            )
        )
        incidence = sparse.coo_matrix(
            (data, (rows, column_indices)),
            shape=(int(self.n_sites), self.n_observations),
            dtype=np.float64,
        )
        return incidence.asformat(format)


@dataclass(frozen=True, slots=True)
class SeparatorQuadraticOperatorView:
    """Least-squares normal operator for a quadratic separator problem.

    Obtain this provisional view from
    :attr:`SeparatorFitProblem.quadratic_operator`.  It is available only for
    ``SquaredLoss`` with no scalar penalties.  Hard restrictions may still be
    present; in that case the matrices describe the quadratic objective while
    ``SeparatorFitProblem.bounds`` separately defines feasibility, and the
    unconstrained normal equation need not hold at a constrained optimum.
    """

    observation_graph: SeparatorObservationGraphView
    observation_rhs: np.ndarray
    regularized_normal_rhs: np.ndarray
    regularization_strength: float
    regularization_reference: np.ndarray
    bounds: PowerFitBounds
    has_hard_constraints: bool

    @property
    def represents_full_quadratic_objective(self) -> bool:
        """Whether this view represents every smooth objective term."""

        return True

    @property
    def normal_equations_characterize_fit(self) -> bool:
        """Whether an unconstrained solver result must satisfy ``A @ w == b``."""

        return not self.has_hard_constraints

    @property
    def observation_nullity(self) -> int:
        """Nullity of the observation Laplacian."""

        return int(self.observation_graph.informative_graph.n_components)

    @property
    def regularized_normal_nullity(self) -> int:
        """Nullity after adding the configured L2 regularization."""

        if float(self.regularization_strength) > 0.0:
            return 0
        return self.observation_nullity

    @property
    def component_alignment_policy(self) -> str:
        """Policy used by the current solver for free component offsets."""

        return self.observation_graph.connectivity.gauge_policy

    @property
    def relative_component_offsets_identified_by_data(self) -> bool:
        """Whether separator data connect every site component."""

        return (
            self.observation_graph.relative_component_offsets_identified_by_data
        )

    @property
    def regularization_removes_observation_nullspace(self) -> bool:
        """Whether positive L2 regularization removes Laplacian null modes."""

        return bool(
            float(self.regularization_strength) > 0.0
            and self.observation_nullity > 0
        )

    def observation_laplacian_matvec(self, vector: np.ndarray) -> np.ndarray:
        """Apply ``L_obs = B @ diag(rho) @ B.T`` without forming a matrix."""

        graph = self.observation_graph
        value = _vector(vector, int(graph.n_sites), name='vector')
        edge_value = graph.rho * (
            value[graph.site_i] - value[graph.site_j]
        )
        result = np.zeros(int(graph.n_sites), dtype=np.float64)
        np.add.at(result, graph.site_i, edge_value)
        np.add.at(result, graph.site_j, -edge_value)
        return result

    def regularized_normal_matvec(self, vector: np.ndarray) -> np.ndarray:
        """Apply ``A = L_obs + regularization_strength * I``."""

        graph = self.observation_graph
        value = _vector(vector, int(graph.n_sites), name='vector')
        result = self.observation_laplacian_matvec(value)
        strength = float(self.regularization_strength)
        if strength > 0.0:
            result += strength * value
        return result

    def observation_laplacian_dense(self) -> np.ndarray:
        """Return ``L_obs`` as a dense NumPy array."""

        graph = self.observation_graph
        n_sites = int(graph.n_sites)
        matrix = np.zeros((n_sites, n_sites), dtype=np.float64)
        np.add.at(matrix, (graph.site_i, graph.site_i), graph.rho)
        np.add.at(matrix, (graph.site_j, graph.site_j), graph.rho)
        np.add.at(matrix, (graph.site_i, graph.site_j), -graph.rho)
        np.add.at(matrix, (graph.site_j, graph.site_i), -graph.rho)
        return matrix

    def regularized_normal_matrix_dense(self) -> np.ndarray:
        """Return ``A = L_obs + regularization_strength * I`` as NumPy."""

        matrix = self.observation_laplacian_dense()
        strength = float(self.regularization_strength)
        if strength > 0.0:
            matrix += strength * np.eye(matrix.shape[0], dtype=np.float64)
        return matrix

    def observation_laplacian_sparse(self, *, format: str = 'csr') -> object:
        """Return ``L_obs`` in a requested optional SciPy sparse format."""

        sparse = _scipy_sparse()
        graph = self.observation_graph
        incidence = graph.incidence_sparse(format='csc')
        edge_weights = sparse.diags(graph.rho, format='csc')
        matrix = incidence @ edge_weights @ incidence.T
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix.asformat(format)

    def regularized_normal_matrix_sparse(
        self,
        *,
        format: str = 'csr',
    ) -> object:
        """Return ``A`` in a requested optional SciPy sparse format."""

        sparse = _scipy_sparse()
        matrix = self.observation_laplacian_sparse(format=format)
        strength = float(self.regularization_strength)
        if strength > 0.0:
            identity = sparse.eye(
                int(self.observation_graph.n_sites),
                dtype=np.float64,
                format=format,
            )
            matrix = matrix + strength * identity
        return matrix.asformat(format)


__all__ = [
    'SeparatorObservationGraphView',
    'SeparatorQuadraticOperatorView',
]
