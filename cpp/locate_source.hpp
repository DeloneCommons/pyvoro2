#ifndef PYVORO2_LOCATE_SOURCE_HPP
#define PYVORO2_LOCATE_SOURCE_HPP

// WP8: observe storage around the unchanged find_voronoi_cell invocation.
// No cell computation, alternate owner selection, or geometry modification.
#include "native_preconditions.hpp"
#include <pybind11/stl.h>
#include <array>
#include <cfloat>
#include <climits>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace pyvoro2::locate_source {
namespace py = pybind11;
using Points = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IDs = py::array_t<int, py::array::c_style | py::array::forcecast>;

[[noreturn]] inline void fail(const char* code, const char* stage,
                             const std::string& reason, int query = -1) {
  throw py::value_error(std::string("locate_native:") + stage + ":" +
      (query < 0 ? "None" : std::to_string(query)) + ":" + code + ":" + reason);
}

inline int step(double value, int query = -1) {
  // A quarter-range leaves room for Div/Mod products and adjacent-block sums.
  if (!std::isfinite(value) || std::abs(value) >= INT_MAX / 4.0)
    fail("LOCATE_NATIVE_UNSUPPORTED", "native", "integer_range", query);
  return value < 0 ? int(value) - 1 : int(value);
}
inline int div(int a, int n) { return a >= 0 ? a / n : -1 + (a + 1) / n; }
inline int mod(int a, int n) { return a >= 0 ? a % n : n - 1 - (n - 1 - a) % n; }
inline bool bits(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}
inline void profile(bool requested) {
  if (!requested) return;
  if (FLT_EVAL_METHOD != 0 || sizeof(int) * CHAR_BIT != 32)
    fail("LOCATE_NATIVE_UNSUPPORTED", "profile", "evaluation_width");
  try { native_preconditions::require_binary64_interval_environment(); }
  catch (const std::exception&) {
    fail("LOCATE_NATIVE_UNSUPPORTED", "profile", "binary64_environment");
  }
}

inline void resources(py::ssize_t n, py::ssize_t m) {
  // Complete integrity-observer allocation, including hash nodes and scratch.
  constexpr std::uint64_t limit = std::uint64_t{64} << 20;
  if (n > (limit - 4096) / 128 || m > (limit - 4096) / 16 ||
      128 * std::uint64_t(n) + 16 * std::uint64_t(m) + 4096 > limit)
    fail("LOCATE_CERTIFICATION_RESOURCE", "insertion", "observer_memory");
}

template<int D> struct State {
  py::array_t<double> stored;
  py::array_t<int> insertion;
  py::array_t<int> query_removals;
  std::array<std::array<double, D>, D> lattice{};
  std::array<int, D> copy_bounds{};
  std::array<double, D> reciprocal{}, width{}, low{};
  std::array<int, D> blocks{};
  std::array<bool, D> periodic{};
  std::vector<int> expected_blocks;
  int storage_blocks = 0;
  State(py::ssize_t n, py::ssize_t m)
      : stored({n, py::ssize_t(D)}), insertion({n, py::ssize_t(D)}),
        query_removals({m, py::ssize_t(D)}) {}
  py::dict packet() const {
    py::dict out;
    out["schema"] = "locate-source-v1";
    out["stored"] = stored;
    out["insertion_shifts"] = insertion;
    out["query_removals"] = query_removals;
    out["lattice"] = lattice;
    out["copy_bounds"] = copy_bounds;
    out["fp_policy"] = "binary64-noncontracting-v1";
    return out;
  }
};

template<int D, class C>
void verify_population(C& con, const Points& points, const IDs& ids,
                       const Points* radii, State<D>& state,
                       const std::vector<int>& expected_blocks,
                       int count) {
  const auto n = points.shape(0);
  auto labels = ids.template unchecked<1>();
  std::unordered_map<int, py::ssize_t> rows;
  for (py::ssize_t i = 0; i < n; ++i) rows.emplace(labels(i), i);
  if (rows.size() != static_cast<std::size_t>(n))
    fail("LOCATE_BACKEND_INSERTION", "insertion", "noninjective_internal_ids");
  std::vector<bool> seen(n, false);
  auto expected = state.stored.template mutable_unchecked<2>();
  for (int block = 0; block < count; ++block) {
    for (int slot = 0; slot < con.co[block]; ++slot) {
      auto it = rows.find(con.id[block][slot]);
      if (it == rows.end() || seen[it->second])
        fail("LOCATE_BACKEND_INSERTION", "insertion", "unexpected_storage_id");
      const auto i = it->second;
      if (expected_blocks[i] != block)
        fail("LOCATE_BACKEND_INSERTION", "insertion", "storage_block_mismatch");
      for (int k = 0; k < D; ++k) {
        const double actual = con.p[block][con.ps * slot + k];
        if (!bits(actual, expected(i, k)))
          fail("LOCATE_BACKEND_INSERTION", "insertion", "storage_operand_mismatch");
        expected(i, k) = actual;
      }
      if (radii && !bits(con.p[block][con.ps * slot + D], radii->data()[i]))
        fail("LOCATE_BACKEND_INSERTION", "insertion", "storage_radius_mismatch");
      seen[i] = true;
    }
  }
  for (bool present : seen) if (!present)
    fail("LOCATE_BACKEND_INSERTION", "insertion", "persistent_input_omitted");
}

template<int D, class C>
State<D> rectangle(C& con, const Points& points, const IDs& ids,
                   const Points* radii, const Points& queries,
                   const std::array<std::array<double, 2>, D>& bounds,
                   const std::array<int, D>& blocks,
                   const std::array<bool, D>& periodic, bool certificate) {
  profile(certificate);
  resources(points.shape(0), queries.shape(0));
  State<D> state(points.shape(0), queries.shape(0));
  state.blocks = blocks; state.periodic = periodic;
  state.width[0] = con.boxx; state.width[1] = con.boxy;
  state.reciprocal[0] = con.xsp; state.reciprocal[1] = con.ysp;
  if constexpr (D == 3) {
    state.width[2] = con.boxz; state.reciprocal[2] = con.zsp;
  }
  auto supplied = points.template unchecked<2>();
  auto stored = state.stored.template mutable_unchecked<2>();
  auto h = state.insertion.template mutable_unchecked<2>();
  std::vector<int> expected(points.shape(0));
  for (int k = 0; k < D; ++k) {
    state.low[k] = bounds[k][0];
    state.lattice[k][k] = bounds[k][1] - bounds[k][0];
    if (blocks[k] >= INT_MAX / 16)
      fail("LOCATE_NATIVE_UNSUPPORTED", "native", "grid_integer_range");
  }
  for (py::ssize_t i = 0; i < points.shape(0); ++i) {
    int block = 0, stride = 1;
    for (int k = 0; k < D; ++k) {
      double value = supplied(i, k);
      int raw = step((value - state.low[k]) * state.reciprocal[k]);
      int mapped = raw; h(i, k) = 0;
      if (periodic[k]) {
        mapped = mod(raw, blocks[k]);
        value += state.width[k] * (mapped - raw);
        h(i, k) = (raw - mapped) / blocks[k];
      } else if (raw < 0 || raw >= blocks[k]) {
        fail("LOCATE_BACKEND_INSERTION", "insertion", "persistent_input_omitted");
      }
      stored(i, k) = value;
      block += stride * mapped; stride *= blocks[k];
    }
    expected[i] = block;
  }
  int count = 1; for (int n : blocks) count *= n;
  state.expected_blocks = std::move(expected); state.storage_blocks = count;
  return state;
}

// Bound every side/vertical copy coefficient from the admitted image grid.
// Two times a rounded absolute expression plus eight is an outward bound on
// the relevant Step/Div operations (inputs are nonnegative and finite).
inline int bound(double value) {
  if (!std::isfinite(value) || value < 0 || value >= INT_MAX / 64.0)
    fail("LOCATE_NATIVE_UNSUPPORTED", "native", "image_integer_range");
  return int(std::ceil(2 * value)) + 8;
}

template<class C>
State<3> triclinic(C& con, const Points& points, const IDs& ids,
                    const Points* radii, const Points& queries, bool certificate) {
  profile(certificate);
  resources(points.shape(0), queries.shape(0));
  State<3> state(points.shape(0), queries.shape(0));
  state.blocks = {con.nx, con.ny, con.nz};
  state.periodic = {true, true, true};
  state.width = {con.boxx, con.boxy, con.boxz};
  state.reciprocal = {con.xsp, con.ysp, con.zsp};
  state.lattice = {{{con.bx, 0, 0}, {con.bxy, con.by, 0},
                    {con.bxz, con.byz, con.bz}}};
  for (int n : {con.nx, con.ny, con.nz, con.ey, con.ez, con.oy, con.oz})
    if (n >= INT_MAX / 64)
      fail("LOCATE_NATIVE_UNSUPPORTED", "native", "grid_integer_range");
  const int z = bound(double(con.oz + con.ez) / con.nz + 1);
  const int y = bound((con.oy + con.ey +
                       bound(z * std::abs(con.byz) * con.ysp)) / double(con.ny) + 1);
  const int x = bound((con.nx + bound((z * std::abs(con.bxz) +
                         (y + 1) * std::abs(con.bxy)) * con.xsp)) / double(con.nx) + 2);
  state.copy_bounds = {x + 2, y + 1, z};
  auto supplied = points.template unchecked<2>();
  auto stored = state.stored.template mutable_unchecked<2>();
  auto h = state.insertion.template mutable_unchecked<2>();
  std::vector<int> expected(points.shape(0));
  for (py::ssize_t i = 0; i < points.shape(0); ++i) {
    std::array<double, 3> p{supplied(i, 0), supplied(i, 1), supplied(i, 2)};
    std::array<int, 3> c{};
    for (int k = 2; k >= 0; --k) {
      c[k] = step(p[k] * state.reciprocal[k]); h(i, k) = 0;
      if (c[k] < 0 || c[k] >= state.blocks[k]) {
        h(i, k) = div(c[k], state.blocks[k]);
        for (int j = k; j >= 0; --j) p[j] -= h(i, k) * state.lattice[k][j];
        c[k] -= h(i, k) * state.blocks[k];
      }
    }
    expected[i] = c[0] + con.nx * (c[1] + con.ey + con.oy * (c[2] + con.ez));
    for (int k = 0; k < 3; ++k) stored(i, k) = p[k];
  }
  state.expected_blocks = std::move(expected); state.storage_blocks = con.oxyz;
  return state;
}

template<int D>
void guard_query(State<D>& state, const Points& queries, int index, bool triclinic) {
  auto q = queries.template unchecked<2>();
  auto a = state.query_removals.template mutable_unchecked<2>();
  std::array<double, D> value{};
  for (int k = 0; k < D; ++k) { value[k] = q(index, k); a(index, k) = 0; }
  for (int order = 0; order < D; ++order) {
    const int k = triclinic ? D - 1 - order : order;
    int c = step((value[k] - state.low[k]) * state.reciprocal[k], index);
    if (c < 0 || c >= state.blocks[k]) {
      if (!state.periodic[k]) return;  // The actual source returns false here.
      a(index, k) = div(c, state.blocks[k]);
      if (triclinic) {
        for (int j = k; j >= 0; --j) value[j] -= a(index, k) * state.lattice[k][j];
      } else value[k] -= a(index, k) * state.lattice[k][k];
      c -= a(index, k) * state.blocks[k];
    }
    // find_voronoi_cell converts this worklist subindex to int before indexing.
    // Native high-end reflection clamps to zero; negative int indices are unsafe.
    const double local = triclinic ? value[k] - state.width[k] * c :
        value[k] - state.low[k] - state.width[k] * c;
    const double work = local * state.reciprocal[k] * 8;
    if (!std::isfinite(work) || work <= -1 || work >= INT_MAX / 4.0)
      fail("LOCATE_NATIVE_UNSUPPORTED", "native", "worklist_index_range", index);
  }
}
}  // namespace pyvoro2::locate_source
#endif
