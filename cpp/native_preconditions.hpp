#ifndef PYVORO2_NATIVE_PRECONDITIONS_HPP
#define PYVORO2_NATIVE_PRECONDITIONS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error "native duplicate certification cannot be built with fast-math"
#endif

namespace pyvoro2::native_preconditions {

namespace py = pybind11;

constexpr std::size_t eager_allocation_limit_bytes = std::size_t{1} << 30;
constexpr double backend_safety_distance_squared = 1e-10;
constexpr double backend_safety_distance = 1e-5;
constexpr std::size_t maximum_safety_candidate_comparisons = 10000000;
constexpr std::size_t maximum_triclinic_safety_candidates = 1000000;
using SafetyBucketIndex = std::int64_t;

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
struct BucketKeyHash {
  std::size_t operator()(
      const std::array<SafetyBucketIndex, Dim>& key) const noexcept {
    std::size_t value = 1469598103934665603ull;
    for (const SafetyBucketIndex component : key) {
      value ^= static_cast<std::size_t>(
          static_cast<std::uint64_t>(component) + 0x9e3779b9ull);
      value *= 1099511628211ull;
    }
    return value;
  }
};

template <std::size_t Dim, class Callback>
inline void visit_neighbor_keys(
                                const std::array<SafetyBucketIndex, Dim>& key,
                                const std::array<SafetyBucketIndex, Dim>& bins,
                                const std::array<bool, Dim>& periodic,
                                Callback&& callback) {
  using Key = std::array<SafetyBucketIndex, Dim>;
  std::unordered_set<Key, BucketKeyHash<Dim>> seen;
  Key current{};
  std::function<void(std::size_t)> visit_axis = [&](std::size_t axis) {
    if (axis == Dim) {
      if (seen.insert(current).second) callback(current);
      return;
    }
    for (SafetyBucketIndex offset = -1; offset <= 1; ++offset) {
      SafetyBucketIndex value = key[axis] + offset;
      if (periodic[axis]) {
        value %= bins[axis];
        if (value < 0) value += bins[axis];
      } else if (value < 0 || value >= bins[axis]) {
        continue;
      }
      current[axis] = value;
      visit_axis(axis + 1);
    }
  };
  visit_axis(0);
}

template <std::size_t Dim>
inline void require_primary_containment(
    const py::array& values,
    const std::array<std::array<double, 2>, Dim>& bounds,
    const std::string& name) {
  const auto* data = static_cast<const double*>(values.data());
  for (py::ssize_t row = 0; row < values.shape(0); ++row) {
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      const double value = data[row * static_cast<py::ssize_t>(Dim) +
                                static_cast<py::ssize_t>(axis)];
      if (!(value >= bounds[axis][0] && value < bounds[axis][1])) {
        fail(name + "[" + std::to_string(row) + "][" +
                 std::to_string(axis) + "]",
             "must lie in the primary half-open interval [" +
                 std::to_string(bounds[axis][0]) + ", " +
                 std::to_string(bounds[axis][1]) + ")");
      }
    }
  }
}

inline SafetyBucketIndex checked_safety_bucket_index(
    long double value,
    const std::string& name) {
  const long double signed_lower_inclusive = -std::ldexp(1.0L, 63);
  const long double signed_upper_exclusive = std::ldexp(1.0L, 63);
  if (!std::isfinite(value) || value < signed_lower_inclusive ||
      value >= signed_upper_exclusive) {
    fail(name, "must fit the half-open signed 64-bit range");
  }
  return static_cast<SafetyBucketIndex>(value);
}

inline bool safety_bucket_index_is_exact_in_wide(SafetyBucketIndex value) {
  constexpr int wide_integer_bits =
      std::numeric_limits<long double>::digits < 63
          ? std::numeric_limits<long double>::digits
          : 63;
  const std::uint64_t magnitude =
      value < 0
          ? static_cast<std::uint64_t>(-(value + 1)) + std::uint64_t{1}
          : static_cast<std::uint64_t>(value);
  const std::uint64_t exact_limit = std::uint64_t{1} << wide_integer_bits;
  return magnitude <= exact_limit;
}

inline SafetyBucketIndex safety_bin_count(long double span,
                                          long double radius,
                                          const std::string& name) {
  if (!(span > 0.0L) || !std::isfinite(span) ||
      !(radius > 0.0L) || !std::isfinite(radius)) {
    fail(name, "requires a positive finite span and radius");
  }
  const long double ratio = span / radius;
  const long double signed_upper_exclusive = std::ldexp(1.0L, 63);
  if (!std::isfinite(ratio) || ratio < 0.0L ||
      ratio >= signed_upper_exclusive) {
    fail(name, "requires more sparse bins than the native key can represent");
  }
  const long double floored = std::floor(ratio);
  if (!std::isfinite(floored) || floored < 0.0L ||
      floored >= signed_upper_exclusive) {
    fail(name, "requires more sparse bins than the native key can represent");
  }
  SafetyBucketIndex count = std::max<SafetyBucketIndex>(
      1, checked_safety_bucket_index(floored, name));
  // Step one bin wider when possible, then verify the width. This deliberately
  // absorbs quotient rounding: the fixed neighboring-bin stencil must never
  // see a cell narrower than the requested radius.
  if (count > 1) --count;
  if (!safety_bucket_index_is_exact_in_wide(count)) {
    fail(name,
         "requires a sparse bin count that the native wide type cannot "
         "represent exactly");
  }
  while (count > 1 && span / static_cast<long double>(count) < radius) {
    --count;
  }
  return count;
}

template <std::size_t Dim>
struct RectangularSafetyLayout {
  std::array<SafetyBucketIndex, Dim> bins{};
  std::array<long double, Dim> widths{};
};

template <std::size_t Dim>
inline RectangularSafetyLayout<Dim> rectangular_safety_layout(
    const std::array<std::array<double, 2>, Dim>& bounds) {
  RectangularSafetyLayout<Dim> layout;
  for (std::size_t axis = 0; axis < Dim; ++axis) {
    const long double span = static_cast<long double>(bounds[axis][1]) -
                             bounds[axis][0];
    layout.bins[axis] = safety_bin_count(
        span, static_cast<long double>(backend_safety_distance),
        "rectangular duplicate candidate layout");
    layout.widths[axis] = std::nextafter(
        span / static_cast<long double>(layout.bins[axis]),
        std::numeric_limits<long double>::infinity());
  }
  return layout;
}

template <std::size_t Dim>
inline std::array<SafetyBucketIndex, Dim> rectangular_safety_key(
    const double* point,
    const std::array<std::array<double, 2>, Dim>& bounds,
    const RectangularSafetyLayout<Dim>& layout) {
  std::array<SafetyBucketIndex, Dim> key{};
  for (std::size_t axis = 0; axis < Dim; ++axis) {
    const long double offset = static_cast<long double>(point[axis]) -
                               bounds[axis][0];
    const long double raw = std::floor(offset / layout.widths[axis]);
    if (!std::isfinite(raw)) {
      fail("rectangular duplicate candidate key", "must be finite");
    }
    SafetyBucketIndex value = raw <= 0.0L
                                  ? 0
                                  : checked_safety_bucket_index(
                                        raw,
                                        "rectangular duplicate candidate key");
    value = std::max<SafetyBucketIndex>(
        0, std::min(value, layout.bins[axis] - 1));
    key[axis] = value;
  }
  return key;
}

template <std::size_t Dim>
inline bool rectangular_pair_is_unsafe(
    const double* left,
    const double* right,
    const std::array<std::array<double, 2>, Dim>& bounds,
    const std::array<bool, Dim>& periodic) {
  // Build a downward binary64 bound for the exact squared distance. This is
  // intentionally conservative and does not depend on long double having more
  // precision than double (MSVC and Apple ARM64 do not provide that).
  double distance_squared_lower = 0.0;
  for (std::size_t axis = 0; axis < Dim; ++axis) {
    const double difference = std::abs(left[axis] - right[axis]);
    if (!std::isfinite(difference)) return true;
    double difference_lower = std::nextafter(difference, 0.0);
    if (periodic[axis]) {
      const double span = bounds[axis][1] - bounds[axis][0];
      if (!std::isfinite(span)) return true;
      const double span_lower = std::nextafter(
          span, -std::numeric_limits<double>::infinity());
      const double difference_upper = std::nextafter(
          difference, std::numeric_limits<double>::infinity());
      const double wrapped = span_lower - difference_upper;
      if (!std::isfinite(wrapped)) return true;
      const double wrapped_lower = std::max(
          0.0, std::nextafter(
                   wrapped, -std::numeric_limits<double>::infinity()));
      difference_lower = std::min(difference_lower, wrapped_lower);
    }
    double square_lower = difference_lower * difference_lower;
    if (!std::isfinite(square_lower)) return true;
    square_lower = std::max(0.0, std::nextafter(square_lower, 0.0));
    const double sum = distance_squared_lower + square_lower;
    if (!std::isfinite(sum)) return true;
    distance_squared_lower = std::max(
        0.0, std::nextafter(sum, -std::numeric_limits<double>::infinity()));
    if (distance_squared_lower > backend_safety_distance_squared) {
      return false;
    }
  }
  return distance_squared_lower <= backend_safety_distance_squared;
}

inline void count_safety_candidate(std::size_t& candidate_count) {
  if (candidate_count == maximum_safety_candidate_comparisons) {
    fail("native duplicate candidate scan",
         "exceeds the private comparison budget of " +
             std::to_string(maximum_safety_candidate_comparisons));
  }
  ++candidate_count;
}

template <std::size_t Dim>
inline std::size_t require_rectangular_duplicate_safety(
    const py::array& points,
    const std::array<std::array<double, 2>, Dim>& bounds,
    const std::array<bool, Dim>& periodic,
    const py::array* inserted_queries) {
  using Key = std::array<SafetyBucketIndex, Dim>;
  const int point_count = static_cast<int>(points.shape(0));
  const int query_count = inserted_queries == nullptr
                              ? 0
                              : static_cast<int>(inserted_queries->shape(0));
  if (point_count < 2 && (point_count == 0 || query_count == 0)) return 0;

  const auto layout = rectangular_safety_layout(bounds);
  std::unordered_map<Key, std::vector<int>, BucketKeyHash<Dim>> buckets;
  const auto* point_data = static_cast<const double*>(points.data());
  std::size_t candidate_count = 0;

  for (int i = 0; i < point_count; ++i) {
    const double* point = point_data + static_cast<std::size_t>(i) * Dim;
    const Key key = rectangular_safety_key(point, bounds, layout);
    visit_neighbor_keys<Dim>(key, layout.bins, periodic, [&](const Key& neighbor) {
      const auto found = buckets.find(neighbor);
      if (found == buckets.end()) return;
      for (const int j : found->second) {
        count_safety_candidate(candidate_count);
        const double* other =
            point_data + static_cast<std::size_t>(j) * Dim;
        if (rectangular_pair_is_unsafe(point, other, bounds, periodic)) {
          fail("points", "contain a backend-unsafe pair at indices " +
                             std::to_string(j) + " and " +
                             std::to_string(i));
        }
      }
    });
    buckets[key].push_back(i);
  }

  if (inserted_queries == nullptr) return candidate_count;
  const auto* query_data =
      static_cast<const double*>(inserted_queries->data());
  for (int query_index = 0; query_index < query_count; ++query_index) {
    const double* query =
        query_data + static_cast<std::size_t>(query_index) * Dim;
    const Key key = rectangular_safety_key(query, bounds, layout);
    visit_neighbor_keys<Dim>(key, layout.bins, periodic, [&](const Key& neighbor) {
      const auto found = buckets.find(neighbor);
      if (found == buckets.end()) return;
      for (const int point_index : found->second) {
        count_safety_candidate(candidate_count);
        const double* point =
            point_data + static_cast<std::size_t>(point_index) * Dim;
        if (rectangular_pair_is_unsafe(query, point, bounds, periodic)) {
          fail("queries", "contain a backend-unsafe ghost at query index " +
                              std::to_string(query_index) +
                              " relative to point index " +
                              std::to_string(point_index));
        }
      }
    });
  }
  return candidate_count;
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
    bool queries_are_inserted = false,
    bool reserve_ghost_id = false) {
  require_positive_controls(blocks.data(), Dim, init_mem);
  validate_arrays<Dim>(points, ids, radii, queries, ghost_radii,
                       reserve_ghost_id);
  validate_box_geometry(bounds, blocks, periodic);
  require_primary_containment(points, bounds, "points");
  const py::array* inserted_queries = queries_are_inserted ? queries : nullptr;
  if (inserted_queries != nullptr) {
    require_primary_containment(*inserted_queries, bounds, "queries");
  }
  require_rectangular_duplicate_safety(points, bounds, periodic,
                                       inserted_queries);

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
  const long double floored_wide = std::floor(scaled);
  if (!std::isfinite(floored_wide) || floored_wide < 0.0L ||
      floored_wide >
          static_cast<long double>(std::numeric_limits<int>::max() - 1)) {
    fail(name, "must fit the C++ int destination range");
  }
  const int floored = static_cast<int>(floored_wide);
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

struct Binary64Interval {
  double lower;
  double upper;
};

static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<double>::digits == 53 &&
                  std::numeric_limits<double>::has_denorm ==
                      std::denorm_present,
              "native duplicate certification requires IEC binary64 double");

inline void require_binary64_interval_environment() {
  if (std::fegetround() != FE_TONEAREST) {
    fail("native duplicate certification",
         "requires round-to-nearest binary64 arithmetic");
  }
  volatile double minimum = std::numeric_limits<double>::denorm_min();
  volatile double two = 2.0;
  volatile double doubled = minimum * two;
  if (!(doubled > minimum)) {
    fail("native duplicate certification",
         "requires gradual binary64 underflow without flush-to-zero");
  }
}

inline bool binary64_interval_is_valid(const Binary64Interval& value) {
  return std::isfinite(value.lower) && std::isfinite(value.upper) &&
         value.lower <= value.upper;
}

inline bool binary64_interval_is_zero(const Binary64Interval& value) {
  return value.lower == 0.0 && value.upper == 0.0;
}

inline Binary64Interval binary64_singleton(double value) {
  return Binary64Interval{value, value};
}

inline double rounded_binary64(double value) {
  // Force every interval primitive through binary64 storage even on targets
  // whose expression evaluator otherwise retains excess precision.
  volatile double stored = value;
  return stored;
}

inline bool binary64_add(const Binary64Interval& lhs,
                         const Binary64Interval& rhs,
                         Binary64Interval& result) {
  if (!binary64_interval_is_valid(lhs) ||
      !binary64_interval_is_valid(rhs)) {
    return false;
  }
  if (binary64_interval_is_zero(lhs)) {
    result = rhs;
    return true;
  }
  if (binary64_interval_is_zero(rhs)) {
    result = lhs;
    return true;
  }
  const double lower = rounded_binary64(lhs.lower + rhs.lower);
  const double upper = rounded_binary64(lhs.upper + rhs.upper);
  if (!std::isfinite(lower) || !std::isfinite(upper)) return false;
  result = {
      std::nextafter(lower, -std::numeric_limits<double>::infinity()),
      std::nextafter(upper, std::numeric_limits<double>::infinity())};
  return binary64_interval_is_valid(result);
}

inline bool binary64_negate(const Binary64Interval& value,
                            Binary64Interval& result) {
  if (!binary64_interval_is_valid(value)) return false;
  result = Binary64Interval{-value.upper, -value.lower};
  return binary64_interval_is_valid(result);
}

inline bool binary64_subtract(const Binary64Interval& lhs,
                              const Binary64Interval& rhs,
                              Binary64Interval& result) {
  Binary64Interval negative{};
  return binary64_negate(rhs, negative) &&
         binary64_add(lhs, negative, result);
}

inline bool binary64_multiply(const Binary64Interval& lhs,
                              const Binary64Interval& rhs,
                              Binary64Interval& result) {
  if (!binary64_interval_is_valid(lhs) ||
      !binary64_interval_is_valid(rhs)) {
    return false;
  }
  if (binary64_interval_is_zero(lhs) || binary64_interval_is_zero(rhs)) {
    result = Binary64Interval{0.0, 0.0};
    return true;
  }
  const std::array<double, 4> products{
      rounded_binary64(lhs.lower * rhs.lower),
      rounded_binary64(lhs.lower * rhs.upper),
      rounded_binary64(lhs.upper * rhs.lower),
      rounded_binary64(lhs.upper * rhs.upper),
  };
  for (const double product : products) {
    if (!std::isfinite(product)) return false;
  }
  const auto endpoints = std::minmax_element(products.begin(), products.end());
  result = {
      std::nextafter(*endpoints.first,
                     -std::numeric_limits<double>::infinity()),
      std::nextafter(*endpoints.second,
                     std::numeric_limits<double>::infinity())};
  return binary64_interval_is_valid(result);
}

inline bool binary64_divide(const Binary64Interval& numerator,
                            const Binary64Interval& denominator,
                            Binary64Interval& result) {
  if (!binary64_interval_is_valid(numerator) ||
      !binary64_interval_is_valid(denominator) ||
      (denominator.lower <= 0.0 && denominator.upper >= 0.0)) {
    return false;
  }
  const std::array<double, 4> quotients{
      rounded_binary64(numerator.lower / denominator.lower),
      rounded_binary64(numerator.lower / denominator.upper),
      rounded_binary64(numerator.upper / denominator.lower),
      rounded_binary64(numerator.upper / denominator.upper),
  };
  for (const double quotient : quotients) {
    if (!std::isfinite(quotient)) return false;
  }
  const auto endpoints =
      std::minmax_element(quotients.begin(), quotients.end());
  result = {
      std::nextafter(*endpoints.first,
                     -std::numeric_limits<double>::infinity()),
      std::nextafter(*endpoints.second,
                     std::numeric_limits<double>::infinity())};
  return binary64_interval_is_valid(result);
}

inline double binary64_abs_upper(const Binary64Interval& value) {
  if (!binary64_interval_is_valid(value)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::max(std::abs(value.lower), std::abs(value.upper));
}

inline bool binary64_square_lower(const Binary64Interval& value,
                                  double& lower) {
  if (!binary64_interval_is_valid(value)) return false;
  if (value.lower <= 0.0 && value.upper >= 0.0) {
    lower = 0.0;
    return true;
  }
  const double magnitude =
      std::min(std::abs(value.lower), std::abs(value.upper));
  const double square = rounded_binary64(magnitude * magnitude);
  if (!std::isfinite(square)) return false;
  lower = std::max(
      0.0, std::nextafter(square,
                          -std::numeric_limits<double>::infinity()));
  return true;
}

struct PeriodicSafetyGeometry {
  std::array<std::array<double, 3>, 3> lattice{};
  std::array<std::array<Binary64Interval, 3>, 3> inverse{};
  std::array<double, 3> coefficient_bound_upper{};
  std::array<SafetyBucketIndex, 3> bins{};
};

inline SafetyBucketIndex certified_periodic_bin_count(
    double coefficient_bound_upper) {
  constexpr double maximum_exact_binary64_integer = 0x1p53;
  constexpr SafetyBucketIndex maximum_exact_bin_count =
      SafetyBucketIndex{1} << 53;
  if (!(coefficient_bound_upper > 0.0) ||
      !std::isfinite(coefficient_bound_upper)) {
    fail("triclinic duplicate candidate layout",
         "requires a finite positive certified coefficient bound");
  }
  if (coefficient_bound_upper >= 1.0) return 1;
  if (coefficient_bound_upper <= 0x1p-53) {
    return maximum_exact_bin_count;
  }
  Binary64Interval quotient{};
  if (!binary64_divide(binary64_singleton(1.0),
                       binary64_singleton(coefficient_bound_upper),
                       quotient)) {
    fail("triclinic duplicate candidate layout",
         "cannot certify a finite reciprocal coefficient bound");
  }
  const double floored = std::floor(quotient.lower);
  if (!std::isfinite(floored) || floored < 1.0) return 1;
  return static_cast<SafetyBucketIndex>(
      std::min(floored, maximum_exact_binary64_integer));
}

inline PeriodicSafetyGeometry periodic_safety_geometry(
    const std::array<double, 6>& params) {
  require_binary64_interval_environment();
  PeriodicSafetyGeometry geometry;
  const Binary64Interval bx = binary64_singleton(params[0]);
  const Binary64Interval bxy = binary64_singleton(params[1]);
  const Binary64Interval by = binary64_singleton(params[2]);
  const Binary64Interval bxz = binary64_singleton(params[3]);
  const Binary64Interval byz = binary64_singleton(params[4]);
  const Binary64Interval bz = binary64_singleton(params[5]);
  const Binary64Interval one = binary64_singleton(1.0);
  geometry.lattice = {{{params[0], 0.0, 0.0},
                       {params[1], params[2], 0.0},
                       {params[3], params[4], params[5]}}};

  Binary64Interval bxy_over_by{};
  Binary64Interval byz_over_by{};
  Binary64Interval bxz_over_bx{};
  Binary64Interval byz_inverse_10{};
  Binary64Interval inverse_20_sum{};
  Binary64Interval negative{};
  if (!binary64_divide(one, bx, geometry.inverse[0][0]) ||
      !binary64_divide(one, by, geometry.inverse[1][1]) ||
      !binary64_divide(one, bz, geometry.inverse[2][2]) ||
      !binary64_multiply(bxy, geometry.inverse[1][1], bxy_over_by) ||
      !binary64_multiply(bxy_over_by, geometry.inverse[0][0], negative) ||
      !binary64_negate(negative, geometry.inverse[1][0]) ||
      !binary64_multiply(byz, geometry.inverse[1][1], byz_over_by) ||
      !binary64_multiply(byz_over_by, geometry.inverse[2][2], negative) ||
      !binary64_negate(negative, geometry.inverse[2][1]) ||
      !binary64_multiply(bxz, geometry.inverse[0][0], bxz_over_bx) ||
      !binary64_multiply(byz, geometry.inverse[1][0], byz_inverse_10) ||
      !binary64_add(bxz_over_bx, byz_inverse_10, inverse_20_sum) ||
      !binary64_negate(inverse_20_sum, negative) ||
      !binary64_multiply(negative, geometry.inverse[2][2],
                         geometry.inverse[2][0])) {
    fail("triclinic duplicate candidate geometry",
         "cannot certify the source-binary64 inverse basis");
  }

  for (std::size_t column = 0; column < 3; ++column) {
    Binary64Interval coefficient_sum = binary64_singleton(0.0);
    for (std::size_t row = 0; row < 3; ++row) {
      const Binary64Interval magnitude = binary64_singleton(
          binary64_abs_upper(geometry.inverse[row][column]));
      Binary64Interval updated{};
      if (!binary64_interval_is_valid(magnitude) ||
          !binary64_add(coefficient_sum, magnitude, updated)) {
        fail("triclinic duplicate candidate geometry",
             "cannot certify finite inverse-basis coefficient bounds");
      }
      coefficient_sum = updated;
    }
    Binary64Interval bound{};
    if (!binary64_multiply(binary64_singleton(backend_safety_distance),
                           coefficient_sum, bound) ||
        !(bound.upper > 0.0)) {
      fail("triclinic duplicate candidate geometry",
           "cannot certify finite positive coefficient bounds");
    }
    geometry.coefficient_bound_upper[column] = bound.upper;
    geometry.bins[column] = certified_periodic_bin_count(bound.upper);
  }
  return geometry;
}

inline void require_periodic_primary_containment(
    const py::array& values,
    const std::array<double, 6>& params,
    const std::string& name) {
  const std::array<double, 3> upper{params[0], params[2], params[5]};
  const auto* data = static_cast<const double*>(values.data());
  for (py::ssize_t row = 0; row < values.shape(0); ++row) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double value = data[row * 3 + static_cast<py::ssize_t>(axis)];
      if (!(value >= 0.0 && value < upper[axis])) {
        fail(name + "[" + std::to_string(row) + "][" +
                 std::to_string(axis) + "]",
             "must lie in the primary half-open interval [0, " +
                 std::to_string(upper[axis]) + ")");
      }
    }
  }
}

using PeriodicSafetyKey = std::array<SafetyBucketIndex, 3>;

inline SafetyBucketIndex positive_modulo(SafetyBucketIndex value,
                                         SafetyBucketIndex modulus) {
  const SafetyBucketIndex remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}

inline bool periodic_coefficient_interval(
    const double* point,
    std::size_t column,
    const PeriodicSafetyGeometry& geometry,
    Binary64Interval& coefficient) {
  coefficient = binary64_singleton(0.0);
  for (std::size_t row = 0; row < 3; ++row) {
    Binary64Interval product{};
    Binary64Interval updated{};
    if (!binary64_multiply(binary64_singleton(point[row]),
                           geometry.inverse[row][column], product) ||
        !binary64_add(coefficient, product, updated)) {
      return false;
    }
    coefficient = updated;
  }
  return true;
}

inline std::vector<PeriodicSafetyKey> periodic_safety_keys(
    const double* point,
    const PeriodicSafetyGeometry& geometry) {
  std::array<std::vector<SafetyBucketIndex>, 3> axis_keys;
  std::size_t key_count = 1;
  for (std::size_t column = 0; column < 3; ++column) {
    Binary64Interval coefficient{};
    Binary64Interval scaled{};
    if (!periodic_coefficient_interval(point, column, geometry,
                                       coefficient) ||
        !binary64_multiply(
            coefficient,
            binary64_singleton(static_cast<double>(geometry.bins[column])),
            scaled)) {
      fail("triclinic duplicate candidate key",
           "cannot certify a finite source-binary64 key interval");
    }
    const double lo = std::floor(scaled.lower);
    const double hi = std::floor(scaled.upper);
    constexpr double minimum_exact_integer = -0x1p53;
    constexpr double maximum_exact_integer = 0x1p53;
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi ||
        lo < minimum_exact_integer || hi > maximum_exact_integer) {
      fail("triclinic duplicate candidate key",
           "requires an integer boundary outside the exact binary64 range");
    }
    const SafetyBucketIndex lower = static_cast<SafetyBucketIndex>(lo);
    const SafetyBucketIndex upper = static_cast<SafetyBucketIndex>(hi);
    const std::uint64_t width =
        static_cast<std::uint64_t>(upper - lower) + std::uint64_t{1};
    if (width > maximum_triclinic_safety_candidates ||
        key_count > maximum_triclinic_safety_candidates / width) {
      fail("triclinic duplicate candidate key",
           "exceeds the private candidate-enumeration budget of " +
               std::to_string(maximum_triclinic_safety_candidates));
    }
    key_count *= static_cast<std::size_t>(width);
    for (SafetyBucketIndex raw = lower;; ++raw) {
      axis_keys[column].push_back(
          positive_modulo(raw, geometry.bins[column]));
      if (raw == upper) break;
    }
    std::sort(axis_keys[column].begin(), axis_keys[column].end());
    axis_keys[column].erase(
        std::unique(axis_keys[column].begin(), axis_keys[column].end()),
        axis_keys[column].end());
  }

  std::vector<PeriodicSafetyKey> keys;
  keys.reserve(key_count);
  for (const SafetyBucketIndex first : axis_keys[0]) {
    for (const SafetyBucketIndex second : axis_keys[1]) {
      for (const SafetyBucketIndex third : axis_keys[2]) {
        keys.push_back(PeriodicSafetyKey{first, second, third});
      }
    }
  }
  return keys;
}

inline std::vector<PeriodicSafetyKey> periodic_safety_neighbor_keys(
    const std::vector<PeriodicSafetyKey>& keys,
    const PeriodicSafetyGeometry& geometry) {
  const std::array<bool, 3> periodic{true, true, true};
  std::unordered_set<PeriodicSafetyKey, BucketKeyHash<3>> unique;
  std::size_t visited = 0;
  for (const PeriodicSafetyKey& key : keys) {
    visit_neighbor_keys<3>(
        key, geometry.bins, periodic,
        [&](const PeriodicSafetyKey& neighbor) {
          if (visited == maximum_triclinic_safety_candidates) {
            fail("triclinic duplicate candidate neighbors",
                 "exceed the private candidate-enumeration budget of " +
                     std::to_string(
                         maximum_triclinic_safety_candidates));
          }
          ++visited;
          unique.insert(neighbor);
        });
  }
  std::vector<PeriodicSafetyKey> neighbors(unique.begin(), unique.end());
  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

inline void count_triclinic_key_aliases(
    std::size_t key_count,
    std::size_t& cumulative_aliases) {
  if (key_count == 0) {
    fail("triclinic duplicate candidate key",
         "must produce at least one certified key");
  }
  const std::size_t aliases = key_count - 1;
  if (aliases > maximum_triclinic_safety_candidates -
                    cumulative_aliases) {
    fail("triclinic duplicate candidate key aliases",
         "exceed the private candidate-enumeration budget of " +
             std::to_string(maximum_triclinic_safety_candidates));
  }
  cumulative_aliases += aliases;
}

inline bool exact_difference_interval(double right,
                                      double left,
                                      Binary64Interval& result) {
  return binary64_subtract(binary64_singleton(right),
                           binary64_singleton(left), result);
}

inline bool add_exact_scaled_value(Binary64Interval& value,
                                   SafetyBucketIndex scale,
                                   double factor) {
  constexpr SafetyBucketIndex maximum_exact_integer =
      SafetyBucketIndex{1} << 53;
  if (scale < -maximum_exact_integer || scale > maximum_exact_integer) {
    return false;
  }
  Binary64Interval product{};
  Binary64Interval updated{};
  if (!binary64_multiply(binary64_singleton(static_cast<double>(scale)),
                         binary64_singleton(factor), product) ||
      !binary64_add(value, product, updated)) {
    return false;
  }
  value = updated;
  return true;
}

inline bool periodic_pair_is_unsafe(
    const double* left,
    const double* right,
    const PeriodicSafetyGeometry& geometry) {
  std::array<Binary64Interval, 3> delta{};
  std::array<Binary64Interval, 3> coefficient{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (!exact_difference_interval(right[axis], left[axis], delta[axis])) {
      fail("triclinic duplicate shift enumeration",
           "cannot certify a finite source-binary64 point difference");
    }
  }
  for (std::size_t column = 0; column < 3; ++column) {
    coefficient[column] = binary64_singleton(0.0);
    for (std::size_t row = 0; row < 3; ++row) {
      Binary64Interval product{};
      Binary64Interval updated{};
      if (!binary64_multiply(delta[row], geometry.inverse[row][column],
                             product) ||
          !binary64_add(coefficient[column], product, updated)) {
        fail("triclinic duplicate shift enumeration",
             "cannot certify a finite inverse-basis coefficient interval");
      }
      coefficient[column] = updated;
    }
  }

  std::array<SafetyBucketIndex, 3> lower{};
  std::array<SafetyBucketIndex, 3> upper{};
  std::size_t candidate_count = 1;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    Binary64Interval negative{};
    Binary64Interval shift_interval{};
    const Binary64Interval radius{
        -geometry.coefficient_bound_upper[axis],
        geometry.coefficient_bound_upper[axis]};
    if (!binary64_negate(coefficient[axis], negative) ||
        !binary64_add(negative, radius, shift_interval)) {
      fail("triclinic duplicate shift enumeration",
           "cannot certify a finite integer-shift interval");
    }
    const double lo = std::ceil(shift_interval.lower);
    const double hi = std::floor(shift_interval.upper);
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
      fail("triclinic duplicate shift enumeration",
           "cannot certify finite integer-shift endpoints");
    }
    if (lo > hi) return false;
    constexpr double minimum_exact_integer = -0x1p53;
    constexpr double maximum_exact_integer = 0x1p53;
    if (lo < minimum_exact_integer || lo > maximum_exact_integer ||
        hi < minimum_exact_integer || hi > maximum_exact_integer) {
      fail("triclinic duplicate shift enumeration",
           "requires shifts outside the exact binary64 integer range");
    }
    const double width_value = hi - lo + 1.0;
    if (!std::isfinite(width_value) ||
        width_value > maximum_triclinic_safety_candidates) {
      fail("triclinic duplicate shift enumeration",
           "exceeds the private candidate-enumeration budget of " +
               std::to_string(maximum_triclinic_safety_candidates));
    }
    lower[axis] = checked_safety_bucket_index(
        lo, "triclinic duplicate shift lower bound");
    upper[axis] = checked_safety_bucket_index(
        hi, "triclinic duplicate shift upper bound");
    const auto width = static_cast<std::size_t>(width_value);
    if (candidate_count > maximum_triclinic_safety_candidates / width) {
      fail("triclinic duplicate shift enumeration",
           "exceeds the private candidate-enumeration budget of " +
               std::to_string(maximum_triclinic_safety_candidates));
    }
    candidate_count *= static_cast<std::size_t>(width);
  }

  const double threshold = backend_safety_distance_squared;
  for (SafetyBucketIndex s0 = lower[0]; s0 <= upper[0]; ++s0) {
    for (SafetyBucketIndex s1 = lower[1]; s1 <= upper[1]; ++s1) {
      for (SafetyBucketIndex s2 = lower[2]; s2 <= upper[2]; ++s2) {
        const std::array<SafetyBucketIndex, 3> shift{s0, s1, s2};
        double squared_lower = 0.0;
        for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
          Binary64Interval value{};
          if (!exact_difference_interval(
                  right[coordinate], left[coordinate], value)) {
            fail("triclinic duplicate distance certification",
                 "cannot certify a finite source-binary64 point difference");
          }
          for (std::size_t lattice_row = 0; lattice_row < 3;
               ++lattice_row) {
            if (!add_exact_scaled_value(
                    value, shift[lattice_row],
                    geometry.lattice[lattice_row][coordinate])) {
              fail("triclinic duplicate distance certification",
                   "cannot certify a shifted source-binary64 coordinate");
            }
          }
          double component_lower = 0.0;
          if (!binary64_square_lower(value, component_lower)) {
            fail("triclinic duplicate distance certification",
                 "cannot certify a finite squared-coordinate lower bound");
          }
          Binary64Interval sum{};
          if (!binary64_add(binary64_singleton(squared_lower),
                            binary64_singleton(component_lower), sum)) {
            fail("triclinic duplicate distance certification",
                 "cannot certify a finite squared-distance lower bound");
          }
          squared_lower = std::max(0.0, sum.lower);
          if (squared_lower > threshold) break;
        }
        if (squared_lower <= threshold) return true;
        if (s2 == std::numeric_limits<SafetyBucketIndex>::max()) break;
      }
      if (s1 == std::numeric_limits<SafetyBucketIndex>::max()) break;
    }
    if (s0 == std::numeric_limits<SafetyBucketIndex>::max()) break;
  }
  return false;
}

inline std::size_t require_periodic_duplicate_safety(
    const py::array& points,
    const std::array<double, 6>& params,
    const py::array* inserted_queries) {
  using Key = std::array<SafetyBucketIndex, 3>;
  const int point_count = static_cast<int>(points.shape(0));
  const int query_count = inserted_queries == nullptr
                              ? 0
                              : static_cast<int>(inserted_queries->shape(0));
  if (point_count < 2 && (point_count == 0 || query_count == 0)) return 0;

  const PeriodicSafetyGeometry geometry = periodic_safety_geometry(params);
  std::unordered_map<Key, std::vector<int>, BucketKeyHash<3>> buckets;
  const auto* point_data = static_cast<const double*>(points.data());
  std::size_t candidate_count = 0;
  std::size_t cumulative_key_aliases = 0;
  for (int i = 0; i < point_count; ++i) {
    const double* point = point_data + static_cast<std::size_t>(i) * 3;
    const std::vector<Key> keys = periodic_safety_keys(point, geometry);
    count_triclinic_key_aliases(keys.size(), cumulative_key_aliases);
    const std::vector<Key> neighbors =
        periodic_safety_neighbor_keys(keys, geometry);
    std::unordered_set<int> candidate_set;
    for (const Key& neighbor : neighbors) {
      const auto found = buckets.find(neighbor);
      if (found == buckets.end()) continue;
      for (const int index : found->second) {
        if (candidate_set.insert(index).second &&
            candidate_set.size() > maximum_safety_candidate_comparisons -
                                       candidate_count) {
          fail("native duplicate candidate scan",
               "exceeds the private comparison budget of " +
                   std::to_string(
                       maximum_safety_candidate_comparisons));
        }
      }
    }
    std::vector<int> candidates(candidate_set.begin(), candidate_set.end());
    std::sort(candidates.begin(), candidates.end());
    for (const int j : candidates) {
      count_safety_candidate(candidate_count);
      const double* other = point_data + static_cast<std::size_t>(j) * 3;
      if (periodic_pair_is_unsafe(point, other, geometry)) {
        fail("points", "contain a backend-unsafe periodic pair at indices " +
                           std::to_string(j) + " and " +
                           std::to_string(i));
      }
    }
    for (const Key& key : keys) {
      auto& indices = buckets[key];
      if (indices.empty() || indices.back() != i) indices.push_back(i);
    }
  }

  if (inserted_queries == nullptr) return candidate_count;
  const auto* query_data =
      static_cast<const double*>(inserted_queries->data());
  for (int query_index = 0; query_index < query_count; ++query_index) {
    const double* query =
        query_data + static_cast<std::size_t>(query_index) * 3;
    const std::vector<Key> keys = periodic_safety_keys(query, geometry);
    count_triclinic_key_aliases(keys.size(), cumulative_key_aliases);
    const std::vector<Key> neighbors =
        periodic_safety_neighbor_keys(keys, geometry);
    std::unordered_set<int> candidate_set;
    for (const Key& neighbor : neighbors) {
      const auto found = buckets.find(neighbor);
      if (found == buckets.end()) continue;
      for (const int index : found->second) {
        if (candidate_set.insert(index).second &&
            candidate_set.size() > maximum_safety_candidate_comparisons -
                                       candidate_count) {
          fail("native duplicate candidate scan",
               "exceeds the private comparison budget of " +
                   std::to_string(
                       maximum_safety_candidate_comparisons));
        }
      }
    }
    std::vector<int> candidates(candidate_set.begin(), candidate_set.end());
    std::sort(candidates.begin(), candidates.end());
    for (const int point_index : candidates) {
      count_safety_candidate(candidate_count);
      const double* point =
          point_data + static_cast<std::size_t>(point_index) * 3;
      if (periodic_pair_is_unsafe(query, point, geometry)) {
        fail("queries",
             "contain a backend-unsafe periodic ghost at query index " +
                 std::to_string(query_index) +
                 " relative to point index " +
                 std::to_string(point_index));
      }
    }
  }
  return candidate_count;
}

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
    const py::array* ghost_radii = nullptr,
    bool queries_are_inserted = false,
    bool reserve_ghost_id = false) {
  require_positive_controls(blocks.data(), blocks.size(), init_mem);
  validate_arrays<3>(points, ids, radii, queries, ghost_radii,
                     reserve_ghost_id);

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

  require_periodic_primary_containment(points, params, "points");
  const py::array* inserted_queries = queries_are_inserted ? queries : nullptr;
  if (inserted_queries != nullptr) {
    require_periodic_primary_containment(*inserted_queries, params, "queries");
  }
  require_periodic_duplicate_safety(points, params, inserted_queries);

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
