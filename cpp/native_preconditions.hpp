#ifndef PYVORO2_NATIVE_PRECONDITIONS_HPP
#define PYVORO2_NATIVE_PRECONDITIONS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

namespace pyvoro2::native_preconditions {

namespace py = pybind11;

constexpr std::size_t eager_allocation_limit_bytes = std::size_t{1} << 30;

[[noreturn]] inline void fail(const std::string& name,
                              const std::string& detail) {
  throw py::value_error(name + " " + detail);
}

inline std::size_t checked_add(std::size_t lhs,
                               std::size_t rhs,
                               const std::string& name) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    fail(name, "overflows the native size range");
  }
  return lhs + rhs;
}

inline std::size_t checked_multiply(std::size_t lhs,
                                    std::size_t rhs,
                                    const std::string& name) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    fail(name, "overflows the native size range");
  }
  return lhs * rhs;
}

inline int checked_int(py::ssize_t value, const std::string& name) {
  if (value < 0 ||
      value > static_cast<py::ssize_t>(std::numeric_limits<int>::max())) {
    fail(name, "must fit the C++ int destination range");
  }
  return static_cast<int>(value);
}

inline int checked_int_add(int lhs, int rhs, const std::string& name) {
  if (lhs < 0 || rhs < 0) {
    fail(name, "requires non-negative operands");
  }
  const std::size_t result = checked_add(static_cast<std::size_t>(lhs),
                                         static_cast<std::size_t>(rhs), name);
  if (result > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    fail(name, "must fit the C++ int destination range");
  }
  return static_cast<int>(result);
}

inline int checked_int_multiply(int lhs, int rhs, const std::string& name) {
  if (lhs < 0 || rhs < 0) {
    fail(name, "requires non-negative operands");
  }
  const std::size_t result = checked_multiply(
      static_cast<std::size_t>(lhs), static_cast<std::size_t>(rhs), name);
  if (result > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    fail(name, "must fit the C++ int destination range");
  }
  return static_cast<int>(result);
}

class ByteEstimate {
 public:
  void add_slots(std::size_t count,
                 std::size_t slot_size,
                 const std::string& name) {
    total_ = checked_add(
        total_, checked_multiply(count, slot_size, name), "byte estimate");
  }

  void add_bytes(std::size_t bytes, const std::string& name) {
    total_ = checked_add(total_, bytes, name);
  }

  std::size_t total() const { return total_; }

  void enforce_limit() const {
    // The frozen policy admits an estimate equal to the cap and rejects only
    // estimates strictly greater than it.
    if (total_ > eager_allocation_limit_bytes) {
      fail("known eager native allocation estimate",
           "exceeds the 1073741824-byte safety limit");
    }
  }

 private:
  std::size_t total_ = 0;
};

inline void require_matrix_shape(const py::array& values,
                                 py::ssize_t columns,
                                 const std::string& name) {
  if (values.ndim() != 2 || values.shape(1) != columns) {
    fail(name, "must have shape (n, " + std::to_string(columns) + ")");
  }
}

inline void require_vector_shape(const py::array& values,
                                 py::ssize_t length,
                                 const std::string& name) {
  if (values.ndim() != 1 || values.shape(0) != length) {
    fail(name, "must have shape (n,)");
  }
}

inline void require_finite_array(const py::array& values,
                                 const std::string& name) {
  const auto* data = static_cast<const double*>(values.data());
  for (py::ssize_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(data[i])) {
      fail(name, "must contain only finite values");
    }
  }
}

inline void require_nonnegative_array(const py::array& values,
                                      const std::string& name) {
  const auto* data = static_cast<const double*>(values.data());
  for (py::ssize_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(data[i])) {
      fail(name, "must contain only finite values");
    }
    if (data[i] < 0.0) {
      fail(name, "must contain only non-negative values");
    }
  }
}

inline void require_internal_ids(const py::array& ids,
                                 int count,
                                 bool reserve_ghost_id) {
  require_vector_shape(ids, count, "ids");
  const auto* data = static_cast<const int*>(ids.data());
  std::vector<unsigned char> seen(static_cast<std::size_t>(count), 0);
  for (int i = 0; i < count; ++i) {
    const int value = data[i];
    if (reserve_ghost_id && value == std::numeric_limits<int>::max()) {
      fail("ids", "must not conflict with the reserved ghost ID");
    }
    if (value < 0) {
      fail("ids", "must contain only non-negative internal IDs");
    }
    if (value >= count) {
      fail("ids", "must contain internal IDs in the range [0, n)");
    }
    auto& marker = seen[static_cast<std::size_t>(value)];
    if (marker != 0) {
      fail("ids", "must not contain duplicate internal IDs");
    }
    marker = 1;
  }
}

template <std::size_t Dim>
struct ArrayCounts {
  int points;
  int queries;
};

template <std::size_t Dim>
inline ArrayCounts<Dim> validate_arrays(const py::array& points,
                                        const py::array& ids,
                                        const py::array* radii,
                                        const py::array* queries,
                                        const py::array* ghost_radii,
                                        bool reserve_ghost_id) {
  require_matrix_shape(points, static_cast<py::ssize_t>(Dim), "points");
  const int point_count = checked_int(points.shape(0), "point count");
  require_finite_array(points, "points");
  require_internal_ids(ids, point_count, reserve_ghost_id);

  if (radii != nullptr) {
    require_vector_shape(*radii, points.shape(0), "radii");
    require_nonnegative_array(*radii, "radii");
  }

  int query_count = 0;
  if (queries != nullptr) {
    require_matrix_shape(*queries, static_cast<py::ssize_t>(Dim), "queries");
    query_count = checked_int(queries->shape(0), "query count");
    require_finite_array(*queries, "queries");
  }
  if (ghost_radii != nullptr) {
    if (queries == nullptr) {
      fail("ghost_radii", "requires a query array");
    }
    require_vector_shape(*ghost_radii, queries->shape(0), "ghost_radii");
    require_nonnegative_array(*ghost_radii, "ghost_radii");
  }
  return ArrayCounts<Dim>{point_count, query_count};
}

inline void require_positive_controls(const int* blocks,
                                      std::size_t dim,
                                      int init_mem) {
  for (std::size_t axis = 0; axis < dim; ++axis) {
    if (blocks[axis] <= 0) {
      fail("blocks[" + std::to_string(axis) + "]",
           "must be a positive C++ int");
    }
  }
  if (init_mem <= 0) {
    fail("init_mem", "must be a positive C++ int");
  }
}

inline double checked_nonnegative_double_add(double lhs,
                                             double rhs,
                                             const std::string& name,
                                             const std::string& expression) {
  const double maximum = std::numeric_limits<double>::max();
  if (rhs > maximum - lhs) {
    fail(name, "produce a non-finite Voro++ " + expression);
  }
  return lhs + rhs;
}

inline double checked_nonnegative_double_multiply(
    double lhs,
    double rhs,
    const std::string& name,
    const std::string& expression) {
  const double maximum = std::numeric_limits<double>::max();
  // If rhs is below one, multiplying it by finite lhs cannot overflow and
  // evaluating maximum/rhs could itself overflow. Only divide in the branch
  // where the quotient is representable.
  if (rhs >= 1.0 && lhs > maximum / rhs) {
    fail(name, "produce a non-finite Voro++ " + expression);
  }
  return lhs * rhs;
}

template <std::size_t Dim>
inline double checked_scaled_sum_of_squares(
    const std::array<double, Dim>& values,
    double scale,
    const std::string& name,
    const std::string& expression) {
  double sum = 0.0;
  const double maximum = std::numeric_limits<double>::max();
  for (const double value : values) {
    const double magnitude = std::abs(value);
    const double scaled = checked_nonnegative_double_multiply(
        scale, magnitude, name, expression);
    const double square = checked_nonnegative_double_multiply(
        scaled, scaled, name, expression);
    sum = checked_nonnegative_double_add(sum, square, name, expression);
  }
  return sum;
}

template <std::size_t Dim>
inline void validate_voro_base_arithmetic(
    const std::array<double, Dim>& block_widths,
    const std::string& name) {
  // v_compute.cc:27 and v_compute_2d.cc:26 form the unweighted bxsq sum.
  checked_scaled_sum_of_squares(block_widths, 1.0, name, "bxsq");

  // v_base.cc:31-60 and v_base_2d.cc:32-55 decode each worklist coordinate
  // as a seven-bit value minus 64, giving [-64, 63], and may pass one extra
  // +/-1 to compute_minimum(). Thus ti/tj/tk are in [-65, 64]. The subregion
  // endpoints stay within half a block, so 65*block_width bounds every
  // multiplication and subtraction in compute_minimum(). Proving the sum of
  // those coordinate bounds squared finite proves every intermediate square
  // and the left-to-right squared-distance accumulation finite.
  constexpr double worklist_coordinate_multiplier = 65.0;
  checked_scaled_sum_of_squares(
      block_widths, worklist_coordinate_multiplier, name,
      "worklist squared-distance accumulation");
}

inline void validate_3d_container_max_len_sq(
    const std::array<double, 3>& lengths,
    const std::array<bool, 3>& periodic) {
  // vendor/voro++/src/container.cc:34-35 evaluates this full-domain
  // expression only for the 3D container. Preserve its source order: square
  // each span, apply the per-axis periodic weight, then accumulate left to
  // right. The 2D container_2d.cc:28-38 has no analogous expression.
  double max_len_sq = 0.0;
  for (std::size_t axis = 0; axis < lengths.size(); ++axis) {
    const double square = checked_nonnegative_double_multiply(
        lengths[axis], lengths[axis], "bounds", "squared length");
    const double weighted_square =
        square * (periodic[axis] ? 0.25 : 1.0);
    max_len_sq = checked_nonnegative_double_add(
        max_len_sq, weighted_square, "bounds", "maximum squared length");
  }
  (void)max_len_sq;
}

template <std::size_t Dim>
inline std::array<double, Dim> validate_box_geometry(
    const std::array<std::array<double, 2>, Dim>& bounds,
    const std::array<int, Dim>& blocks,
    const std::array<bool, Dim>& periodic) {
  std::array<double, Dim> lengths{};
  std::array<double, Dim> block_widths{};
  // Common geometry validates only values evaluated by both container_base
  // implementations: finite spans, block widths, and their reciprocals.
  for (std::size_t axis = 0; axis < Dim; ++axis) {
    const double lo = bounds[axis][0];
    const double hi = bounds[axis][1];
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
      fail("bounds[" + std::to_string(axis) + "]",
           "must contain only finite values");
    }
    if (!(hi > lo)) {
      fail("bounds[" + std::to_string(axis) + "]",
           "must be strictly ordered with a finite positive length");
    }
    const long double length_wide = static_cast<long double>(hi) -
                                    static_cast<long double>(lo);
    if (length_wide > static_cast<long double>(
                          std::numeric_limits<double>::max())) {
      fail("bounds[" + std::to_string(axis) + "]",
           "must be strictly ordered with a finite positive length");
    }
    const double length = static_cast<double>(length_wide);
    lengths[axis] = length;

    const double block_width = length / blocks[axis];
    if (!(block_width > 0.0) || !std::isfinite(block_width) ||
        !std::isfinite(1.0 / block_width)) {
      fail("bounds and blocks",
           "produce an unsafe Voro++ block width or reciprocal");
    }
    block_widths[axis] = block_width;
  }

  if constexpr (Dim == 3) {
    validate_3d_container_max_len_sq(lengths, periodic);
  } else {
    static_assert(Dim == 2, "only 2D and 3D boxes are supported");
  }

  // Both backends construct voro_base and voro_compute from block widths.
  validate_voro_base_arithmetic(block_widths, "bounds and blocks");
  return lengths;
}

inline void add_particle_block_storage(ByteEstimate& estimate,
                                       int block_count,
                                       int init_mem,
                                       int particle_stride) {
  const std::size_t blocks = static_cast<std::size_t>(block_count);
  const std::size_t initial = static_cast<std::size_t>(init_mem);
  estimate.add_slots(blocks, sizeof(int*), "container ID pointers");
  estimate.add_slots(blocks, sizeof(double*), "container particle pointers");
  estimate.add_slots(checked_multiply(2, blocks, "container counter slots"),
                     sizeof(int), "container counter arrays");

  const std::size_t per_particle = checked_add(
      sizeof(int),
      checked_multiply(static_cast<std::size_t>(particle_stride),
                       sizeof(double), "particle coordinate bytes"),
      "particle slot bytes");
  const std::size_t slots = checked_multiply(
      blocks, initial, "initial per-block particle slots");
  estimate.add_slots(slots, per_particle, "initial particle storage");
}

inline void add_3d_worklist_storage(ByteEstimate& estimate) {
  // vendor/voro++/src/v_base.cc:23-25 and worklist.hh:27-29.
  estimate.add_slots(64 * 64, sizeof(double), "3D worklist radii");
}

inline void add_2d_worklist_storage(ByteEstimate& estimate) {
  // vendor/voro++/2d/src/v_base_2d.cc:24-26 and worklist_2d.hh:27-29.
  estimate.add_slots(16 * 64, sizeof(double), "2D worklist radii");
}

inline void add_3d_initial_unit_cell_storage(ByteEstimate& estimate) {
  // The periodic unitcell owns one voronoicell. These are the eager arrays in
  // vendor/voro++/src/cell.cc:20-44 using config.hh:19-35.
  constexpr std::size_t vertices = 256;
  constexpr std::size_t vertex_orders = 64;
  estimate.add_slots(vertices, sizeof(int*), "unit-cell edge pointers");
  estimate.add_slots(vertices, sizeof(int), "unit-cell vertex orders");
  estimate.add_slots(vertices, sizeof(unsigned int), "unit-cell mask");
  estimate.add_slots(4 * vertices, sizeof(double), "unit-cell points");
  estimate.add_slots(2 * vertex_orders, sizeof(int), "unit-cell order counters");
  estimate.add_slots(vertex_orders, sizeof(int*), "unit-cell order pointers");
  estimate.add_slots(2 * 256 + 32, sizeof(int), "unit-cell work stacks");
  for (std::size_t order = 0; order < vertex_orders; ++order) {
    const std::size_t count =
        order == 3 ? 256 * 7 : 8 * (2 * order + 1);
    estimate.add_slots(count, sizeof(int), "unit-cell order storage");
  }
}

template <std::size_t Dim>
inline void preflight_box(
    const py::array& points,
    const py::array& ids,
    const py::array* radii,
    const std::array<std::array<double, 2>, Dim>& bounds,
    const std::array<int, Dim>& blocks,
    const std::array<bool, Dim>& periodic,
    int init_mem,
    int particle_stride,
    const py::array* queries = nullptr,
    const py::array* ghost_radii = nullptr,
    bool reserve_ghost_id = false) {
  require_positive_controls(blocks.data(), Dim, init_mem);
  validate_arrays<Dim>(points, ids, radii, queries, ghost_radii,
                       reserve_ghost_id);
  validate_box_geometry(bounds, blocks, periodic);

  checked_int_multiply(particle_stride, init_mem, "ps * init_mem");

  int block_count = 1;
  std::array<int, Dim> mask_dims{};
  for (std::size_t axis = 0; axis < Dim; ++axis) {
    block_count = checked_int_multiply(block_count, blocks[axis],
                                       "rectangular block product");
    if (periodic[axis]) {
      mask_dims[axis] = checked_int_add(
          checked_int_multiply(2, blocks[axis], "periodic mask dimension"),
          1, "periodic mask dimension");
    } else {
      mask_dims[axis] = blocks[axis];
    }
  }

  ByteEstimate estimate;
  add_particle_block_storage(estimate, block_count, init_mem, particle_stride);

  if constexpr (Dim == 3) {
    const int hxy = checked_int_multiply(mask_dims[0], mask_dims[1],
                                         "3D mask plane size");
    const int hxyz = checked_int_multiply(hxy, mask_dims[2],
                                          "3D mask size");
    const int hx_plus_hy = checked_int_add(mask_dims[0], mask_dims[1],
                                           "3D queue dimension sum");
    const int hz_term = checked_int_multiply(mask_dims[2], hx_plus_hy,
                                              "3D queue z term");
    const int queue_inner = checked_int_add(
        checked_int_add(3, hxy, "3D queue length"), hz_term,
        "3D queue length");
    const int queue_size = checked_int_multiply(3, queue_inner,
                                                "3D queue length");
    estimate.add_slots(static_cast<std::size_t>(hxyz), sizeof(unsigned int),
                       "3D compute mask");
    estimate.add_slots(static_cast<std::size_t>(queue_size), sizeof(int),
                       "3D compute queue");
    add_3d_worklist_storage(estimate);
    // vendor/voro++/src/container.cc:519-521 and config.hh:35.
    estimate.add_slots(32, sizeof(void*), "3D wall pointer storage");
  } else {
    static_assert(Dim == 2, "only 2D and 3D boxes are supported");
    const int hxy = checked_int_multiply(mask_dims[0], mask_dims[1],
                                         "2D mask size");
    const int dimension_sum = checked_int_add(mask_dims[0], mask_dims[1],
                                               "2D queue dimension sum");
    const int queue_inner = checked_int_add(2, dimension_sum,
                                             "2D queue length");
    const int queue_size = checked_int_multiply(2, queue_inner,
                                                "2D queue length");
    estimate.add_slots(static_cast<std::size_t>(hxy), sizeof(unsigned int),
                       "2D compute mask");
    estimate.add_slots(static_cast<std::size_t>(queue_size), sizeof(int),
                       "2D compute queue");
    add_2d_worklist_storage(estimate);
    // vendor/voro++/2d/src/container_2d.cc:486-487 and config.hh:17.
    estimate.add_slots(32, sizeof(void*), "2D wall pointer storage");
  }
  estimate.enforce_limit();
}

inline int periodic_extent_bound(long double extent_bound,
                                 int blocks,
                                 double period,
                                 const std::string& name) {
  const long double infinity =
      std::numeric_limits<long double>::infinity();
  long double scaled = std::nextafter(
      extent_bound / static_cast<long double>(period), infinity);
  if (!std::isfinite(scaled) || scaled < 0.0L) {
    fail(name, "must fit the C++ int destination range");
  }
  const long double block_count = static_cast<long double>(blocks);
  if (block_count > 0.0L &&
      scaled > std::numeric_limits<long double>::max() / block_count) {
    fail(name, "must fit the C++ int destination range");
  }
  scaled = std::nextafter(scaled * block_count, infinity);
  if (!std::isfinite(scaled) || scaled < 0.0L ||
      scaled > static_cast<long double>(std::numeric_limits<int>::max() - 1)) {
    fail(name, "must fit the C++ int destination range");
  }
  const int floored = static_cast<int>(std::floor(scaled));
  return checked_int_add(floored, 1, name);
}

inline long double checked_wide_add_up(long double lhs,
                                       long double rhs,
                                       const std::string& name) {
  const long double maximum = std::numeric_limits<long double>::max();
  if (!std::isfinite(lhs) || !std::isfinite(rhs) || lhs < 0.0L ||
      rhs < 0.0L || rhs > maximum - lhs) {
    fail(name, "must be finite in the native wide type");
  }
  const long double upper = std::nextafter(
      lhs + rhs, std::numeric_limits<long double>::infinity());
  if (!std::isfinite(upper)) {
    fail(name, "must be finite in the native wide type");
  }
  return upper;
}

inline long double periodic_source_extent_bound(
    const std::array<double, 6>& params) {
  // unitcell.cc stores doubled cell vertices and halves max(y+norm) and
  // max(z+norm), so its physical source quantities are
  // max_v(v_y+||v||) and max_v(v_z+||v||). For the lower-triangular basis,
  //
  //   ||a|| + ||b|| + ||c||
  //     <= |bx| + |bxy| + by + |bxz| + |byz| + bz = E.
  //
  // The source quantities are at most twice the usual covering-radius bound,
  // namely ||a||+||b||+||c||, so E bounds both required extents. Round every
  // checked positive addition upward, then round the final bound upward once
  // more, so accumulation cannot turn the mathematical upper bound downward.
  long double bound = 0.0L;
  for (const double component : params) {
    bound = checked_wide_add_up(
        bound, std::abs(static_cast<long double>(component)),
        "periodic source-extent bound");
  }
  const long double upper = std::nextafter(
      bound, std::numeric_limits<long double>::infinity());
  if (!std::isfinite(upper)) {
    fail("periodic source-extent bound",
         "must be finite in the native wide type");
  }
  return upper;
}

inline long double periodic_shell_coordinate_bound(
    std::initializer_list<double> components) {
  // unitcell.cc:36 and 44-56 use l < 2*max_unit_voro_shells, where
  // max_unit_voro_shells is 10 (config.hh:93). Using 20 rather than the
  // attained maximum coefficient 19 is a conservative source-derived bound.
  constexpr long double shell_index_bound = 20.0L;
  const long double coordinate_limit = std::sqrt(
      static_cast<long double>(std::numeric_limits<double>::max()));
  long double bound = 0.0L;
  for (const double component : components) {
    const long double magnitude =
        std::abs(static_cast<long double>(component));
    if (magnitude > coordinate_limit / shell_index_bound) {
      fail("periodic cell parameters",
           "produce a non-finite Voro++ unit-cell shell-vector squared norm");
    }
    const long double term = shell_index_bound * magnitude;
    if (term > coordinate_limit - bound) {
      fail("periodic cell parameters",
           "produce a non-finite Voro++ unit-cell shell-vector squared norm");
    }
    bound += term;
  }
  return bound;
}

inline void validate_periodic_shell_arithmetic(
    const std::array<double, 6>& params) {
  // unitcell.cc:91 and 197 form
  //   x=i*bx+j*bxy+k*bxz, y=j*by+k*byz, z=k*bz,
  // followed by x*x+y*y+z*z at line 198 and in cell.hh:374-376. The complete
  // component bounds include every shear term. Accumulating against the
  // remaining DBL_MAX proves each square and each source-order addition before
  // Voro++ evaluates them in binary64.
  const std::array<long double, 3> coordinate_bounds = {
      periodic_shell_coordinate_bound(
          {params[0], params[1], params[3]}),
      periodic_shell_coordinate_bound({params[2], params[4]}),
      periodic_shell_coordinate_bound({params[5]}),
  };
  long double remaining =
      static_cast<long double>(std::numeric_limits<double>::max());
  for (const long double bound : coordinate_bounds) {
    if (bound > std::sqrt(remaining)) {
      fail("periodic cell parameters",
           "produce a non-finite Voro++ unit-cell shell-vector squared norm");
    }
    remaining -= bound * bound;
    if (remaining < 0.0L) {
      fail("periodic cell parameters",
           "produce a non-finite Voro++ unit-cell shell-vector squared norm");
    }
  }

  // The same coefficient bound covers unitcell.cc:32-33 and cell.cc:297:
  // uc*=10 followed by a factor of two in init_base(). Plane clipping forms
  // convex combinations of those initial vertices. Thus the shell proof also
  // bounds constructor-time plane dot products, unitcell.cc:69's vertex norm,
  // and max_radius_squared() (cell.cc:1907-1916); the attained shell
  // coefficient is strictly below 20.
}

inline void validate_periodic_unit_cell_tolerances(
    const std::array<double, 6>& params) {
  const std::array<double, 3> diagonal = {
      params[0], params[2], params[5]};
  const double diagonal_square_sum = checked_scaled_sum_of_squares(
      diagonal, 1.0, "periodic cell parameters",
      "unit-cell diagonal squared sum");

  // unitcell.cc:28 passes 4*max_unit_voro_shells^2 times the diagonal squared
  // sum to voronoicell_base. With max_unit_voro_shells=10, this is 400.
  const double max_len_sq = checked_nonnegative_double_multiply(
      400.0, diagonal_square_sum, "periodic cell parameters",
      "unit-cell maximum squared length");

  // cell.cc:26-27, config.hh:80-82. Check the operations in source order so
  // max_len_sq being finite is not mistaken for a proof that tol_cu is finite.
  const double voro_tolerance =
      10.0 * std::numeric_limits<double>::epsilon();
  const double tol = checked_nonnegative_double_multiply(
      voro_tolerance, max_len_sq, "periodic cell parameters",
      "unit-cell tolerance");
  const double sqrt_tol = std::sqrt(tol);
  checked_nonnegative_double_multiply(
      tol, sqrt_tol, "periodic cell parameters",
      "unit-cell cubic tolerance");
  checked_nonnegative_double_multiply(
      20.0, tol, "periodic cell parameters",
      "unit-cell large tolerance");
}

struct Periodic3DResourceEstimate {
  long double source_extent_bound;
  int primary_blocks;
  int ey_bound;
  int ez_bound;
  int oy;
  int oz;
  int extended_blocks;
  int hx;
  int hy;
  int hz;
  int mask_size;
  int queue_size;
  ByteEstimate known_eager_allocation;
};

inline Periodic3DResourceEstimate estimate_periodic_3d_resources(
    const std::array<double, 6>& params,
    const std::array<int, 3>& blocks,
    int init_mem,
    int particle_stride) {
  checked_int_multiply(particle_stride, init_mem, "ps * init_mem");
  const int primary_xy = checked_int_multiply(blocks[0], blocks[1],
                                               "periodic primary block plane");
  const int primary_blocks = checked_int_multiply(
      primary_xy, blocks[2], "periodic primary block product");

  const long double source_extent_bound =
      periodic_source_extent_bound(params);
  const int ey_bound = periodic_extent_bound(
      source_extent_bound, blocks[1], params[2], "periodic ey bound");
  const int ez_bound = periodic_extent_bound(
      source_extent_bound, blocks[2], params[5], "periodic ez bound");
  const int oy = checked_int_add(
      blocks[1], checked_int_multiply(2, ey_bound, "periodic oy"),
      "periodic oy");
  const int oz = checked_int_add(
      blocks[2], checked_int_multiply(2, ez_bound, "periodic oz"),
      "periodic oz");
  const int extended_xy = checked_int_multiply(
      blocks[0], oy, "periodic extended block plane");
  const int extended_blocks = checked_int_multiply(
      extended_xy, oz, "periodic extended block product");

  const int hx = checked_int_add(
      checked_int_multiply(2, blocks[0], "periodic compute hx"), 1,
      "periodic compute hx");
  const int hy = checked_int_add(
      checked_int_multiply(2, ey_bound, "periodic compute hy"), 1,
      "periodic compute hy");
  const int hz = checked_int_add(
      checked_int_multiply(2, ez_bound, "periodic compute hz"), 1,
      "periodic compute hz");
  const int hxy = checked_int_multiply(hx, hy,
                                       "periodic compute mask plane");
  const int mask_size = checked_int_multiply(
      hxy, hz, "periodic compute mask size");
  const int hx_plus_hy = checked_int_add(hx, hy,
                                         "periodic queue dimension sum");
  const int hz_term = checked_int_multiply(hz, hx_plus_hy,
                                            "periodic queue z term");
  const int queue_inner = checked_int_add(
      checked_int_add(3, hxy, "periodic queue length"), hz_term,
      "periodic queue length");
  const int queue_size = checked_int_multiply(3, queue_inner,
                                              "periodic queue length");

  ByteEstimate estimate;
  const std::size_t extended = static_cast<std::size_t>(extended_blocks);
  // container_prd.cc:34-35: id/p, co/mem, and img span oxyz.
  estimate.add_slots(extended, sizeof(int*), "periodic extended ID pointers");
  estimate.add_slots(extended, sizeof(double*),
                     "periodic extended particle pointers");
  estimate.add_slots(checked_multiply(2, extended,
                                      "periodic extended counter slots"),
                     sizeof(int), "periodic extended counter arrays");
  estimate.add_slots(extended, sizeof(char), "periodic image flags");

  // container_prd.cc:43-49 allocates particle arrays only for primary blocks.
  const std::size_t primary_slots = checked_multiply(
      static_cast<std::size_t>(primary_blocks),
      static_cast<std::size_t>(init_mem), "periodic primary particle slots");
  const std::size_t per_particle = checked_add(
      sizeof(int),
      checked_multiply(static_cast<std::size_t>(particle_stride),
                       sizeof(double), "periodic particle coordinate bytes"),
      "periodic particle slot bytes");
  estimate.add_slots(primary_slots, per_particle,
                     "periodic primary particle storage");
  estimate.add_slots(static_cast<std::size_t>(mask_size), sizeof(unsigned int),
                     "periodic compute mask");
  estimate.add_slots(static_cast<std::size_t>(queue_size), sizeof(int),
                     "periodic compute queue");
  add_3d_worklist_storage(estimate);
  add_3d_initial_unit_cell_storage(estimate);

  return Periodic3DResourceEstimate{
      source_extent_bound,
      primary_blocks,
      ey_bound,
      ez_bound,
      oy,
      oz,
      extended_blocks,
      hx,
      hy,
      hz,
      mask_size,
      queue_size,
      estimate,
  };
}

inline void preflight_periodic_3d(
    const py::array& points,
    const py::array& ids,
    const py::array* radii,
    const std::array<double, 6>& params,
    const std::array<int, 3>& blocks,
    int init_mem,
    int particle_stride,
    const py::array* queries = nullptr,
    const py::array* ghost_radii = nullptr) {
  require_positive_controls(blocks.data(), blocks.size(), init_mem);
  validate_arrays<3>(points, ids, radii, queries, ghost_radii, false);

  for (std::size_t i = 0; i < params.size(); ++i) {
    if (!std::isfinite(params[i])) {
      fail("cell_params[" + std::to_string(i) + "]", "must be finite");
    }
  }
  for (const std::size_t i : {std::size_t{0}, std::size_t{2}, std::size_t{5}}) {
    if (!(params[i] > 0.0)) {
      fail("cell_params[" + std::to_string(i) + "]",
           "must be a positive finite periodic length");
    }
  }

  validate_periodic_shell_arithmetic(params);
  validate_periodic_unit_cell_tolerances(params);

  std::array<double, 3> block_widths{};
  for (std::size_t axis = 0; axis < blocks.size(); ++axis) {
    const double period = params[axis == 0 ? 0 : axis == 1 ? 2 : 5];
    const double block_width = period / blocks[axis];
    if (!(block_width > 0.0) || !std::isfinite(block_width) ||
        !std::isfinite(1.0 / block_width)) {
      fail("cell_params and blocks",
           "produce an unsafe Voro++ block width or reciprocal");
    }
    block_widths[axis] = block_width;
  }
  validate_voro_base_arithmetic(block_widths, "cell_params and blocks");

  const Periodic3DResourceEstimate resources =
      estimate_periodic_3d_resources(params, blocks, init_mem,
                                     particle_stride);
  resources.known_eager_allocation.enforce_limit();
}

}  // namespace pyvoro2::native_preconditions

#endif
