// WP6 ordinary planar occurrence witness.  The clipping/traversal source stays
// byte-unmodified; its existing scalar neighbor lifecycle carries our tokens.
#include "planar_witness.hpp"
#include "planar_witness_contract.hpp"
#include "native_preconditions.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <tuple>
#include <type_traits>
#include "voro++_2d.hh"
// This TU replaces the standalone vendor TU, including its stock explicit
// instantiations, and also instantiates the actual template for our adapter.
#include "v_compute_2d.cc"

namespace pyvoro2::planar_witness {
namespace {
namespace py = pybind11;
using namespace voro;
using Points = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IDs = py::array_t<int, py::array::c_style | py::array::forcecast>;
using Bounds = std::array<std::array<double, 2>, 2>;
using Blocks = std::array<int, 2>;
using Mask = std::array<bool, 2>;
using Options = std::tuple<bool, bool, bool>;

// Bounds on additional Python witness materialization, independent of the
// complete native image family and any exact semantic-audit budget.  Match
// the attribution consumer limit before allocating unusable Python records;
// 262144 expanded occurrence dictionaries are approximately 128 MiB in CPython.
constexpr std::size_t max_occurrences = 262144;
constexpr std::size_t max_sources = 1000000;
constexpr std::size_t max_observer_storage_bytes = 64 * 1024 * 1024;
constexpr int max_token_sources = (INT_MAX - 4) / 9;

static_assert(sizeof(unsigned int) * CHAR_BIT == 32 && UINT_MAX == 4294967295u,
              "ordinary planar source requires 32-bit unsigned int");
static_assert(FLT_EVAL_METHOD == 0, "ordinary planar witness requires binary64 evaluation");
#ifndef PYVORO2_PLANAR_FP_CONTROLLED
#error "ordinary planar witness must use its target FP contract"
#endif

bool qualified_cohort() {
#if defined(__linux__) && defined(__x86_64__) && defined(__SSE2__) && \
    defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 13 && \
    __GNUC_MINOR__ == 3 && !defined(__AVX__) && !defined(__FMA__) && !defined(__FMA4__)
  return true;
#else
  return false;
#endif
}

py::dict profile() {
  const bool source = std::strcmp(PYVORO2_PLANAR_SOURCE_SHA256,
                                  PYVORO2_PLANAR_APPROVED_SHA256) == 0;
  const bool nearest = std::fegetround() == FE_TONEAREST;
  const bool gradual = gradual_underflow();
  py::dict out;
  out["schema"] = PYVORO2_PLANAR_SCHEMA;
  out["source_sha256"] = PYVORO2_PLANAR_SOURCE_SHA256;
  out["build_sha256"] = PYVORO2_PLANAR_BUILD_SHA256;
  out["compiler"] = PYVORO2_PLANAR_COMPILER_ID;
  out["compiler_version"] = PYVORO2_PLANAR_COMPILER_VERSION;
  out["cohort"] = "linux-x86_64-gcc13.3-sse2";
  out["cohort_supported"] = qualified_cohort();
  out["source_supported"] = source;
  out["flt_eval_method"] = FLT_EVAL_METHOD;
  out["int_bits"] = 32;
  out["uint_bits"] = 32;
  out["double_digits"] = std::numeric_limits<double>::digits;
  out["round_to_nearest"] = nearest;
  out["gradual_underflow"] = gradual;
  out["fp_contract"] = "off";
  out["fast_math"] = false;
  out["lto"] = false;
  out["max_sources"] = max_sources;
  out["max_observer_storage_bytes"] = max_observer_storage_bytes;
  out["max_occurrences"] = max_occurrences;
  out["max_token_sources"] = max_token_sources;
  out["qualified"] = source && qualified_cohort() && nearest && gradual;
  return out;
}

void require_profile() {
  require_evaluation();
  if (std::strcmp(PYVORO2_PLANAR_SOURCE_SHA256,
                  PYVORO2_PLANAR_APPROVED_SHA256) != 0)
    fail("profile", "source", "compiled ordinary planar source closure is unqualified");
  if (!qualified_cohort())
    fail("profile", "cohort", "ordinary planar compiler/target cohort is unqualified");
}

bool same_bits(double a, double b) {
  std::uint64_t aa, bb;
  std::memcpy(&aa, &a, sizeof aa);
  std::memcpy(&bb, &b, sizeof bb);
  return aa == bb;
}

struct Insertion {
  std::array<double, 2> point;
  std::array<int, 2> h;
  int block;
  double radius;
};

// The source's Step is trunc(v)-1 for negative v, including negative exact
// integers.  Preserve it literally; this is not mathematical floor.
template<class Container>
Insertion replay_insertion(Container& con, const double* p, double radius) {
  Insertion out{{p[0], p[1]}, {0, 0}, 0, radius};
  const double low[2] = {con.ax, con.ay};
  const double reciprocal[2] = {con.xsp, con.ysp};
  const double width[2] = {con.boxx, con.boxy};
  const int blocks[2] = {con.nx, con.ny};
  const bool periodic[2] = {con.xperiodic, con.yperiodic};
  int mapped[2];
  for (int axis = 0; axis < 2; ++axis) {
    const double quotient = (p[axis] - low[axis]) * reciprocal[axis];
    if (!std::isfinite(quotient) || quotient <= INT_MIN + 1.0 || quotient >= INT_MAX)
      fail("insertion", "quotient", "native insertion integer conversion is outside its checked range");
    const int raw = quotient < 0 ? int(quotient) - 1 : int(quotient);
    if (!periodic[axis]) {
      // Leave omission to the observed population check after actual put().
      mapped[axis] = raw;
    } else {
      const int n = blocks[axis];
      const std::int64_t wide = raw;
      mapped[axis] = raw >= 0 ? raw % n : n - 1 - (n - 1 - wide) % n;
      const std::int64_t delta = std::int64_t(mapped[axis]) - raw;
      if (delta < INT_MIN || delta > INT_MAX)
        fail("insertion", "integer_range", "native insertion coordinate update would overflow");
      out.h[axis] = static_cast<int>(-delta / n);
      out.point[axis] += width[axis] * static_cast<int>(delta);
    }
  }
  // Only form a block for an admitted replay; source put returns first on an
  // omitted nonperiodic axis and never evaluates a later block expression.
  if (mapped[0] < 0 || mapped[0] >= con.nx ||
      mapped[1] < 0 || mapped[1] >= con.ny) out.block = -1;
  else out.block = mapped[0] + con.nx * mapped[1];
  return out;
}

struct State {
  int source = -1, source_block = -1, source_slot = -1;
  int candidate = -1, active_block = -1;
  int sx = 0, sy = 0, count = 0;
};

int decoded_owner(int token, int count) {
  if (token >= 1 && token <= 4) return -token;
  if (token < 5 || (token - 5) / 9 >= count)
    fail("provenance", "token", "unassociated outgoing-edge token");
  return (token - 5) / 9;
}

struct ObservedCell : voronoicell_neighbor_2d {
  State& state;
  explicit ObservedCell(State& value) : state(value) {}
  ObservedCell(const ObservedCell&) = delete;
  ObservedCell& operator=(const ObservedCell&) = delete;

  void init(double xl, double xh, double yl, double yh) {
    voronoicell_neighbor_2d::init(xl, xh, yl, yh);
    for (int i = 0; i < 4; ++i) {
      if (ne[i] < -4 || ne[i] > -1)
        fail("provenance", "initialization", "unknown ordinary initialization side");
      ne[i] = -ne[i];
    }
  }

  void neighbors(std::vector<int>& result) override {
    result.resize(p);
    for (int i = 0; i < p; ++i) result[i] = decoded_owner(ne[i], state.count);
  }

  bool nplane(double x, double y, double rs, int owner) {
    if (state.source < 0 || owner != state.candidate || owner < 0 || owner >= state.count)
      fail("provenance", "candidate", "actual cut owner disagrees with its source hook");
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(rs))
      fail("native", "cut_arithmetic", "native cut operands are nonfinite");
    const int token = 5 + 9 * owner + 3 * (state.sx + 1) + state.sy + 1;
    // Only metadata changes.  The stock nonvirtual clipping method dispatches
    // its original n_set/n_copy/reallocation hooks and never reads labels to
    // decide geometry.  No-op cuts consequently retain the old token.
    return voronoicell_neighbor_2d::nplane(x, y, rs, token);
  }
};

template<class Base>
struct ObservedContainer : Base {
  using Base::r_init;
  using Base::r_ctest;
  using Base::r_cutoff;
  using Base::r_prime;
  using Base::r_max_add;
  using Base::r_current_sub;
  State state;
  voro_compute_2d<ObservedContainer<Base>> producer;
  ObservedContainer(const Bounds& b, const Blocks& n, const Mask& m, int mem)
      : Base(b[0][0], b[0][1], b[1][0], b[1][1], n[0], n[1], m[0], m[1], mem),
        producer(*this, m[0] ? 2 * n[0] + 1 : n[0], m[1] ? 2 * n[1] + 1 : n[1]) {}

  template<class Cell>
  bool initialize_voronoicell(Cell& cell, int ij, int s, int ci, int cj,
                              int& i, int& j, double& x, double& y, int& disp) {
    if (this->wep != this->walls)
      fail("profile", "route", "ordinary witness excludes custom wall objects");
    state.source = this->id[ij][s];
    state.source_block = state.active_block = ij;
    state.source_slot = s;
    state.candidate = -1;
    state.sx = state.sy = 0;
    return Base::initialize_voronoicell(cell, ij, s, ci, cj, i, j, x, y, disp);
  }

  int region_index(int ci, int cj, int ei, int ej, double& qx, double& qy, int& disp) {
    const int hx = this->xperiodic ? 2 * this->nx + 1 : this->nx;
    const int hy = this->yperiodic ? 2 * this->ny + 1 : this->ny;
    if (ci < 0 || ci >= this->nx || cj < 0 || cj >= this->ny ||
        ei < 0 || ei >= hx || ej < 0 || ej >= hy)
      fail("provenance", "mask", "source producer escaped its complete image family");
    state.sx = this->xperiodic ? (ci + ei < this->nx ? -1 : ci + ei >= 2 * this->nx ? 1 : 0) : 0;
    state.sy = this->yperiodic ? (cj + ej < this->ny ? -1 : cj + ej >= 2 * this->ny ? 1 : 0) : 0;
    const int block = Base::region_index(ci, cj, ei, ej, qx, qy, disp);
    const int px = this->xperiodic ? ci + ei - this->nx - state.sx * this->nx : ei;
    const int py = this->yperiodic ? cj + ej - this->ny - state.sy * this->ny : ej;
    if (px < 0 || px >= this->nx || py < 0 || py >= this->ny || block != px + this->nx * py)
      fail("provenance", "region", "source image and actual owner block disagree");
    state.active_block = block;
    return block;
  }

  void candidate(int ij, int slot) {
    if (ij != state.active_block || ij < 0 || ij >= this->nxy ||
        slot < 0 || slot >= this->co[ij])
      fail("provenance", "candidate", "candidate storage is not associated with the active region");
    state.candidate = this->id[ij][slot];
  }
  double r_scale(double value, int ij, int slot) {
    candidate(ij, slot);
    return Base::r_scale(value, ij, slot);
  }
  bool r_scale_check(double& value, double mrs, int ij, int slot) {
    candidate(ij, slot);
    return Base::r_scale_check(value, mrs, ij, slot);
  }
};

void validate_topology(voronoicell_neighbor_2d& cell) {
  if (cell.p <= 0 || cell.p > cell.current_vertices)
    fail("provenance", "topology", "invalid final native vertex count");
  std::vector<unsigned char> seen(static_cast<std::size_t>(cell.p), 0);
  int slot = 0;
  for (int step = 0; step < cell.p; ++step) {
    if (slot < 0 || slot >= cell.p || seen[slot])
      fail("provenance", "topology", "final outgoing slots do not form one native cycle");
    seen[slot] = 1;
    const int next = cell.ed[2 * slot];
    if (next < 0 || next >= cell.p || cell.ed[2 * next + 1] != slot)
      fail("provenance", "topology", "inconsistent outgoing and incoming native slots");
    if (!std::isfinite(cell.pts[2 * slot]) || !std::isfinite(cell.pts[2 * slot + 1]))
      fail("native", "vertices", "final native local coordinates are nonfinite");
    slot = next;
  }
  if (slot != 0) fail("provenance", "topology", "native outgoing cycle is not closed");
}

py::dict serialize_cell(voronoicell_neighbor_2d& cell, int id, double x, double y,
                        const Options& opts) {
  py::dict out;
  out["id"] = id;
  out["area"] = cell.area();
  out["site"] = py::cast(std::vector<double>{x, y});
  if (std::get<0>(opts)) {
    std::vector<double> vertices;
    cell.vertices(x, y, vertices);
    py::list rows;
    for (int i = 0; i < cell.p; ++i)
      rows.append(py::cast(std::vector<double>{vertices[2 * i], vertices[2 * i + 1]}));
    out["vertices"] = rows;
  }
  if (std::get<1>(opts)) {
    py::list rows;
    for (int i = 0; i < cell.p; ++i)
      rows.append(py::cast(std::vector<int>{cell.ed[2 * i], cell.ed[2 * i + 1]}));
    out["adjacency"] = rows;
  }
  if (std::get<2>(opts)) {
    std::vector<int> neighbors;
    cell.neighbors(neighbors);
    py::list edges;
    for (int i = 0; i < cell.p; ++i) {
      py::dict edge;
      edge["adjacent_cell"] = neighbors[i];
      edge["vertices"] = py::cast(std::vector<int>{i, cell.ed[2 * i]});
      edges.append(edge);
    }
    out["edges"] = edges;
  }
  return out;
}

template<class Base, bool Power, bool Observe = true, bool VerifyProfile = true>
py::tuple run(Points points, IDs ids, Points radii, Bounds bounds, Blocks blocks,
              Mask periodic, int init_mem, Options opts) {
  try {
    if constexpr (VerifyProfile) require_profile();
    else require_evaluation();  // Explicitly experimental qualification entry.
    native_preconditions::preflight_box<2>(points, ids, Power ? &radii : nullptr,
                                          bounds, blocks, periodic, init_mem, Power ? 3 : 2);
    const auto count = points.shape(0);
    if (count > max_token_sources)
      fail("resource", "token_range", "complete source token family exceeds signed native int");
    if (static_cast<std::size_t>(count) > max_sources)
      fail("resource", "witness_sources", "source snapshot limit exceeded");
    if constexpr (Observe) {
      // Base owns its ordinary producer.  The adapter adds a second producer;
      // account for its mask, queue and radius worklists before construction.
      const std::size_t hx = periodic[0] ? 2 * std::size_t(blocks[0]) + 1 : blocks[0];
      const std::size_t hy = periodic[1] ? 2 * std::size_t(blocks[1]) + 1 : blocks[1];
      const std::size_t extra = sizeof(unsigned int) * hx * hy +
          sizeof(int) * 2 * (2 + hx + hy) + sizeof(double) * 16 * 64;
      if (extra > max_observer_storage_bytes)
        fail("resource", "observer_storage", "additional native producer storage exceeds the witness budget");
    }
    using Container = std::conditional_t<Observe, ObservedContainer<Base>, Base>;
    std::unique_ptr<Container> storage;
    if constexpr (Observe) storage = std::make_unique<Container>(bounds, blocks, periodic, init_mem);
    else storage = std::make_unique<Container>(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1],
                                              blocks[0], blocks[1], periodic[0], periodic[1], init_mem);
    auto& con = *storage;
    if constexpr (Observe) con.state.count = static_cast<int>(count);
    auto p = points.unchecked<2>();
    auto id = ids.unchecked<1>();
    std::vector<Insertion> replay(static_cast<std::size_t>(count));
    for (py::ssize_t k = 0; k < count; ++k) {
      const double radius = Power ? *radii.data(k) : 0;
      replay[id(k)] = replay_insertion(con, points.data(k), radius);
      if constexpr (Power) con.put(id(k), p(k, 0), p(k, 1), radius);
      else con.put(id(k), p(k, 0), p(k, 1));
    }
    require_population(con, count);
    py::list inserted;
    for (int block = 0; block < con.nxy; ++block) {
      for (int slot = 0; slot < con.co[block]; ++slot) {
        const int pid = con.id[block][slot];
        const auto& expected = replay[pid];
        const double* actual = con.p[block] + con.ps * slot;
        if (expected.block != block || !same_bits(expected.point[0], actual[0]) ||
            !same_bits(expected.point[1], actual[1]) ||
            (Power && !same_bits(expected.radius, actual[2])))
          fail("insertion", "storage", "actual inserted storage disagrees with qualified source replay");
        py::dict row;
        row["id"] = pid;
        row["point"] = py::make_tuple(actual[0], actual[1]);
        row["radius"] = Power ? actual[2] : 0.0;
        row["h"] = expected.h;
        row["block"] = block;
        row["slot"] = slot;
        inserted.append(row);
      }
    }

    using Cell = std::conditional_t<Observe, ObservedCell, voronoicell_neighbor_2d>;
    std::unique_ptr<Cell> cell_storage;
    if constexpr (Observe) cell_storage = std::make_unique<ObservedCell>(con.state);
    else cell_storage = std::make_unique<voronoicell_neighbor_2d>();
    auto& cell = *cell_storage;
    py::list cells, sources;
    std::size_t occurrences = 0;
    c_loop_all_2d loop(con);
    if (loop.start()) do {
      int pid;
      double x, y, radius;
      loop.pos(pid, x, y, radius);
      require_selector(con, x, y, loop.i, loop.j);
      bool present;
      if constexpr (Observe)
        present = con.producer.compute_cell(static_cast<ObservedCell&>(cell), loop.ij, loop.q, loop.i, loop.j);
      else present = con.compute_cell(cell, loop);
      py::dict source;
      source["id"] = pid;
      source["present"] = present;
      py::list local2, next, origins;
      if (present) {
        if (static_cast<std::size_t>(cell.p) > max_occurrences - occurrences)
          fail("resource", "witness_occurrences", "final occurrence materialization limit exceeded");
        occurrences += static_cast<std::size_t>(cell.p);
        validate_topology(cell);
        if constexpr (Observe) {
          if (con.state.source != pid || con.state.source_block != loop.ij || con.state.source_slot != loop.q)
            fail("provenance", "source", "final cell is not associated with its stored source");
        }
        for (int slot = 0; slot < cell.p; ++slot) {
          local2.append(py::make_tuple(cell.pts[2 * slot], cell.pts[2 * slot + 1]));
          next.append(cell.ed[2 * slot]);
          if constexpr (Observe) {
            const int token = cell.ne[slot];
            const int owner = decoded_owner(token, static_cast<int>(count));
            py::dict origin;
            origin["source"] = pid;
            origin["slot"] = slot;
            origin["next"] = cell.ed[2 * slot];
            if (token <= 4) {
              origin["kind"] = "initialization";
              origin["side"] = owner;
            } else {
              const int image = (token - 5) % 9;
              const int sx = image / 3 - 1, sy = image % 3 - 1;
              if ((!periodic[0] && sx) || (!periodic[1] && sy))
                fail("provenance", "image", "token has a nonperiodic image coefficient");
              origin["kind"] = "particle";
              origin["owner"] = owner;
              origin["sigma"] = py::make_tuple(sx, sy);
            }
            origins.append(origin);
          }
        }
        cells.append(serialize_cell(cell, pid, x, y, opts));
      }
      source["local2"] = local2;
      source["next"] = next;
      source["origins"] = origins;
      sources.append(source);
    } while (loop.inc());
    py::dict packet;
    packet["profile"] = profile();
    packet["bounds"] = bounds;
    packet["periods"] = py::make_tuple(con.bx - con.ax, con.by - con.ay);
    packet["periodic"] = periodic;
    packet["inserted"] = inserted;
    packet["sources"] = sources;
    return py::make_tuple(cells, packet);
  } catch (const std::bad_alloc&) {
    fail("resource", "allocation", "native witness allocation failed before completion");
  }
}

template<bool Observe, bool VerifyProfile = true>
void bind_compute(py::module_& m, const char* standard, const char* power) {
  m.def(standard, [](Points p, IDs ids, Bounds b, Blocks n, Mask mask, int mem, Options opts) {
    return run<container_2d, false, Observe, VerifyProfile>(p, ids, Points(), b, n, mask, mem, opts);
  }, py::arg("points"), py::arg("ids"), py::arg("bounds"), py::arg("blocks"),
     py::arg("periodic"), py::arg("init_mem"), py::arg("opts"));
  m.def(power, &run<container_poly_2d, true, Observe, VerifyProfile>, py::arg("points"), py::arg("ids"),
        py::arg("radii"), py::arg("bounds"), py::arg("blocks"), py::arg("periodic"),
        py::arg("init_mem"), py::arg("opts"));
}
}  // namespace

void bind(pybind11::module_& module) {
  module.def("_planar_witness_profile", &profile);
  bind_compute<true>(module, "_compute_box_standard_witness", "_compute_box_power_witness");
#ifdef PYVORO2_PLANAR_QUALIFICATION
  bind_compute<false, false>(module, "_planar_stock_standard", "_planar_stock_power");
  bind_compute<true, false>(module, "_planar_candidate_standard", "_planar_candidate_power");
#endif
}
}  // namespace pyvoro2::planar_witness
