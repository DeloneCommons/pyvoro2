#include "native_witness.hpp"
#include "native_preconditions.hpp"

#include <pybind11/stl.h>

#include <cfloat>
#include <climits>
#include <memory>
#include <new>
#include <type_traits>

// Instantiate a distinct compute type directly from the vendored source. The
// four adapters expose only radius operations which the original compute type
// accesses through friendship. Ordinary producer instantiations remain in
// v_compute.cc; every _core translation unit shares the NativeFP policy.
namespace voro {
template <class Base>
class witness_container_access : public Base {
 public:
  using Base::Base;
  using Base::r_init;
  using Base::r_prime;
  using Base::r_ctest;
  using Base::r_cutoff;
  using Base::r_max_add;
  using Base::r_current_sub;
  using Base::r_scale;
  using Base::r_scale_check;
};
using witness_container = witness_container_access<container>;
using witness_container_poly = witness_container_access<container_poly>;
using witness_container_periodic = witness_container_access<container_periodic>;
using witness_container_periodic_poly =
    witness_container_access<container_periodic_poly>;
using witness_seed_cell = pyvoro2::native_witness::ObservedCell;
}  // namespace voro

#undef VOROPP_V_COMPUTE_HH
#define voro_compute witness_compute
#define particle_record witness_particle_record
#define container witness_container
#define container_poly witness_container_poly
#define container_periodic witness_container_periodic
#define container_periodic_poly witness_container_periodic_poly
#include "v_compute.cc"
#undef container_periodic_poly
#undef container_periodic
#undef container_poly
#undef container
#undef particle_record
#undef voro_compute

// Reuse unitcell's exact shell selection, order, vector arithmetic, and stop
// rule. Its native geometry is never replaced by this provenance replay.
#undef VOROPP_UNITCELL_HH
#define unitcell witness_unitcell
#define voronoicell witness_seed_cell
#include "unitcell.cc"
#undef voronoicell
#undef unitcell

namespace pyvoro2::native_witness {
namespace {
namespace py = pybind11;
namespace pre = native_preconditions;
using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Integers = py::array_t<int, py::array::c_style | py::array::forcecast>;
using Bounds = std::array<std::array<double, 2>, 3>;

void require_environment() {
  pre::require_binary64_interval_environment();
  if (FLT_EVAL_METHOD != 0)
    throw std::runtime_error("native witness requires binary64 expression evaluation");
}

py::dict profile_metadata() {
  py::dict build;
  const auto runtime = native_runtime::inspect();
  build["binary64"] = true;
  build["round_to_nearest"] = runtime.supported && runtime.nearest;
  build["gradual_underflow"] = runtime.supported && runtime.subnormal;
  build["runtime_compatible"] = runtime.compatible();
  build["float_eval_method"] = FLT_EVAL_METHOD;
  build["fast_math"] = false;
  build["fp_contract"] = "off";
  build["fp_contract_scope"] =
      "all _core translation units, including ordinary producer and clipping";
  build["native_fp_policy"] = "binary64-noncontracting-v1";
  build["ipo"] = false;
  build["source_coupled_seed_replay"] = true;
  build["source_coupled_compute"] = true;
  build["compiler_id"] = PYVORO2_WITNESS_COMPILER_ID;
#if defined(__x86_64__) || defined(_M_X64)
  build["x86_64"] = true;
#else
  build["x86_64"] = false;
#endif
#if defined(__SSE2__) || defined(_M_X64)
  build["sse2"] = true;
#else
  build["sse2"] = false;
#endif
#if defined(__AVX__) || defined(_M_AVX)
  build["avx"] = true;
#else
  build["avx"] = false;
#endif
#if defined(__FMA__) || defined(_M_FMA)
  build["fma"] = true;
#else
  build["fma"] = false;
#endif
#ifdef __VERSION__
  build["compiler"] = __VERSION__;
#else
  build["compiler"] = "MSVC " PYVORO2_WITNESS_COMPILER_VERSION;
#endif
  build["source_sha256"] = PYVORO2_WITNESS_SOURCE_SHA256;
  build["source_sha256_scope"] = "listed files; not a complete source fingerprint";
  build["max_unit_voro_shells"] = voro::max_unit_voro_shells;
  build["int_bits"] = sizeof(int) * CHAR_BIT;
  build["int_min"] = std::numeric_limits<int>::min();
  build["int_max"] = std::numeric_limits<int>::max();
  return build;
}

py::dict build_metadata() {
  py::dict build = profile_metadata();
  build["tolerance"] = voro::tolerance;
  build["big_tolerance_factor"] = voro::big_tolerance_fac;
  return build;
}

void ghost_metadata(py::dict& build) {
  build["ghost_selected_route"] = "wp7-initialized-selected-v1";
  build["ghost_source_sha256"] = PYVORO2_GHOST_SOURCE_SHA256;
  build["ghost_source_sha256_scope"] =
      "all vendored 3D source files plus 3D bindings/witness/preconditions/FP guard/CMake/NativeFP/pyproject";
}

template <class Ordinary, class Observer>
void add_grid_context(const Ordinary& ordinary, const Observer& observer,
                      py::dict& context) {
  const std::array<double, 3> widths{
      ordinary.boxx, ordinary.boxy, ordinary.boxz};
  const std::array<double, 3> reciprocals{
      ordinary.xsp, ordinary.ysp, ordinary.zsp};
  const std::array<double, 3> other_widths{
      observer.boxx, observer.boxy, observer.boxz};
  const std::array<double, 3> other_reciprocals{
      observer.xsp, observer.ysp, observer.zsp};
  for (int axis = 0; axis < 3; ++axis)
    if (!same_bits(widths[axis], other_widths[axis]) ||
        !same_bits(reciprocals[axis], other_reciprocals[axis]))
      throw std::runtime_error("native witness noninterference: block operands");
  context["block_widths"] = widths;
  context["block_reciprocals"] = reciprocals;
  if constexpr (std::is_base_of_v<voro::container_periodic_base, Ordinary>) {
    if (ordinary.ey != observer.ey || ordinary.ez != observer.ez ||
        ordinary.wy != observer.wy || ordinary.wz != observer.wz ||
        ordinary.oy != observer.oy || ordinary.oz != observer.oz ||
        ordinary.oxyz != observer.oxyz)
      throw std::runtime_error("native witness noninterference: image grid");
    py::dict grid;
    grid["ey"] = ordinary.ey;
    grid["ez"] = ordinary.ez;
    grid["wy"] = ordinary.wy;
    grid["wz"] = ordinary.wz;
    grid["oy"] = ordinary.oy;
    grid["oz"] = ordinary.oz;
    grid["oxyz"] = ordinary.oxyz;
    context["image_grid"] = grid;
    context["mask_shape"] = std::array<int, 3>{
        2*ordinary.nx+1, 2*ordinary.ey+1, 2*ordinary.ez+1};
  } else {
    context["mask_shape"] = std::array<int, 3>{
        ordinary.xperiodic ? 2*ordinary.nx+1 : ordinary.nx,
        ordinary.yperiodic ? 2*ordinary.ny+1 : ordinary.ny,
        ordinary.zperiodic ? 2*ordinary.nz+1 : ordinary.nz};
  }
}

py::dict origin_dict(const Origin& origin) {
  py::dict out;
  out["token"] = origin.token;
  out["kind"] = origin.kind;
  out["owner"] = origin.owner;
  out["normal"] = origin.normal;
  out["offset"] = origin.offset;
  out["legacy_owner"] = origin.legacy_owner;
  out["axis"] = origin.axis;
  out["sense"] = origin.sense;
  out["periodic"] = origin.periodic;
  out["side"] = origin.side;
  return out;
}

py::dict cell_dict(voro::voronoicell_neighbor& ordinary, ObservedCell& observed,
                   bool computed, int owner, const std::array<double, 3>& site,
                   const py::object& radius) {
  py::dict out;
  out["id"] = owner;
  out["site"] = site;
  out["radius"] = radius;
  out["computed"] = computed;
  if (!same_bits(ordinary.tol, observed.tol) ||
      !same_bits(ordinary.tol_cu, observed.tol_cu) ||
      !same_bits(ordinary.big_tol, observed.big_tol))
    throw std::runtime_error("native witness noninterference: cell tolerance");
  out["tol"] = observed.tol;
  out["tol_cu"] = observed.tol_cu;
  out["big_tol"] = observed.big_tol;
  py::list origins;
  for (const Origin& origin : observed.origins) origins.append(origin_dict(origin));
  out["origins"] = origins;
  py::list vertices, orders, adjacency, faces;
  double volume = 0.0;
  if (computed) {
    require_same_geometry(ordinary, observed);
    volume = ordinary.volume();
    if (!same_bits(volume, observed.volume()))
      throw std::runtime_error("native witness noninterference: volume bits");
    for (int i = 0; i < ordinary.p; ++i) {
      vertices.append(std::array<double, 3>{observed.pts[4*i],
                      observed.pts[4*i+1], observed.pts[4*i+2]});
      orders.append(observed.nu[i]);
      std::vector<int> adjacent;
      for (int j = 0; j < ordinary.nu[i]; ++j) {
        adjacent.push_back(observed.ed[i][j]);
        if (ordinary.ne[i][j] != observed.origin(observed.ne[i][j]).legacy_owner)
          throw std::runtime_error("native witness noninterference: edge owner");
      }
      adjacency.append(adjacent);
    }
    const auto cycles = observed.witness_faces();
    std::vector<int> flat, legacy_owners;
    ordinary.face_vertices(flat);
    ordinary.neighbors(legacy_owners);
    std::size_t cursor = 0;
    if (cycles.size() != legacy_owners.size())
      throw std::runtime_error("native witness noninterference: serialized face count");
    for (std::size_t i = 0; i < cycles.size(); ++i) {
      const auto& cycle = cycles[i];
      if (cursor >= flat.size() || flat[cursor++] !=
              static_cast<int>(cycle.vertices.size()))
        throw std::runtime_error("native witness noninterference: serialized face size");
      for (int vertex : cycle.vertices)
        if (cursor >= flat.size() || flat[cursor++] != vertex)
          throw std::runtime_error("native witness noninterference: serialized face cycle");
      if (cycle.legacy_owner != legacy_owners[i])
        throw std::runtime_error("native witness noninterference: serialized face owner");
      py::dict face;
      face["vertices"] = cycle.vertices;
      face["token"] = cycle.token;
      face["edge_tokens"] = cycle.edge_tokens;
      face["legacy_owner"] = cycle.legacy_owner;
      faces.append(face);
    }
    if (cursor != flat.size())
      throw std::runtime_error("native witness noninterference: serialized trailing data");
  }
  out["volume"] = volume;
  out["vertices_doubled"] = vertices;
  out["vertex_orders"] = orders;
  out["adjacency"] = adjacency;
  out["faces"] = faces;
  py::dict parity;
  parity["computed"] = true;
  for (const char* key : {"geometry", "topology", "owners", "volume"})
    parity[key] = computed ? py::object(py::bool_(true)) : py::object(py::none());
  out["noninterference"] = parity;
  return out;
}

template <class Ordinary, class Observer, class Loop>
py::dict collect(Ordinary& ordinary, Observer& observer,
                 voro::witness_compute<Observer>& compute,
                 const Doubles& points, const Integers& ids,
                 const Doubles* radii, const std::array<bool, 3>& periodic,
                 ObservedCell* seed, py::dict context) {
  auto coordinates = points.unchecked<2>();
  auto identifiers = ids.unchecked<1>();
  const double* radii_data = radii ? radii->data() : nullptr;
  for (py::ssize_t i = 0; i < points.shape(0); ++i) {
    if constexpr (std::is_base_of_v<voro::radius_poly, Ordinary>) {
      ordinary.put(identifiers(i), coordinates(i,0), coordinates(i,1),
                   coordinates(i,2), radii_data[i]);
      observer.put(identifiers(i), coordinates(i,0), coordinates(i,1),
                   coordinates(i,2), radii_data[i]);
    } else {
      ordinary.put(identifiers(i), coordinates(i,0), coordinates(i,1), coordinates(i,2));
      observer.put(identifiers(i), coordinates(i,0), coordinates(i,1), coordinates(i,2));
    }
  }
  Loop original_loop(ordinary), observer_loop(observer);
  voro::voronoicell_neighbor original_cell;
  ObservedCell observed_cell;
  py::list cells, sites;
  std::vector<py::object> stored_sites(static_cast<std::size_t>(points.shape(0)));
  std::vector<unsigned char> seen(stored_sites.size(), 0);
  bool original_more = original_loop.start(), observer_more = observer_loop.start();
  while (original_more || observer_more) {
    if (original_more != observer_more)
      throw std::runtime_error("native witness noninterference: particle count");
    int owner, observed_owner;
    double x, y, z, r, ox, oy, oz, other_radius;
    original_loop.pos(owner, x, y, z, r);
    observer_loop.pos(observed_owner, ox, oy, oz, other_radius);
    if (owner != observed_owner || !same_bits(x, ox) || !same_bits(y, oy) ||
        !same_bits(z, oz) || !same_bits(r, other_radius))
      throw std::runtime_error("native witness noninterference: particle storage");
    if (original_loop.i != observer_loop.i ||
        original_loop.j != observer_loop.j ||
        original_loop.k != observer_loop.k ||
        original_loop.ijk != observer_loop.ijk ||
        original_loop.q != observer_loop.q)
      throw std::runtime_error("native witness noninterference: particle block");
    if (owner < 0 || static_cast<std::size_t>(owner) >= seen.size() || seen[owner])
      throw std::runtime_error("native witness invalid persistent site identity");
    seen[owner] = 1;
    py::object radius = radii ? py::object(py::float_(r)) : py::object(py::none());
    py::dict stored;
    stored["id"] = owner;
    stored["site"] = std::array<double, 3>{x,y,z};
    stored["radius"] = radius;
    stored["block"] = std::array<int, 3>{
        original_loop.i, original_loop.j, original_loop.k};
    stored["block_index"] = original_loop.ijk;
    stored["block_slot"] = original_loop.q;
    stored_sites[owner] = stored;
    if (seed) observed_cell.begin_periodic(owner, *seed);
    else observed_cell.begin_box(owner, periodic);
    const bool original_success = ordinary.compute_cell(original_cell, original_loop);
    const bool observed_success = compute.compute_cell(
        observed_cell, observer_loop.ijk, observer_loop.q,
        observer_loop.i, observer_loop.j, observer_loop.k);
    if (original_success != observed_success)
      throw std::runtime_error("native witness noninterference: computation status");
    cells.append(cell_dict(original_cell, observed_cell, original_success,
                            owner, {x,y,z}, radius));
    original_more = original_loop.inc();
    observer_more = observer_loop.inc();
  }
  for (std::size_t owner = 0; owner < seen.size(); ++owner) {
    if (!seen[owner])
      throw std::runtime_error("native witness missing persistent site identity");
    sites.append(stored_sites[owner]);
  }
  py::dict packet;
  packet["cells"] = cells;
  packet["sites"] = sites;
  packet["context"] = context;
  packet["build"] = build_metadata();
  return packet;
}

template <class Ordinary, class Observer>
py::dict observe_box(const Doubles& points, const Integers& ids,
                     const Bounds& bounds, const std::array<int, 3>& blocks,
                     const std::array<bool, 3>& periodic, int init_mem,
                     const Doubles* radii) {
  auto make = [&](auto type) {
    using Container = typename decltype(type)::type;
    return std::make_unique<Container>(bounds[0][0], bounds[0][1],
        bounds[1][0], bounds[1][1], bounds[2][0], bounds[2][1],
        blocks[0], blocks[1], blocks[2], periodic[0], periodic[1],
        periodic[2], init_mem);
  };
  auto original = make(std::common_type<Ordinary>{});
  auto observer = make(std::common_type<Observer>{});
  voro::witness_compute<Observer> compute(*observer,
      periodic[0] ? 2*blocks[0]+1 : blocks[0],
      periodic[1] ? 2*blocks[1]+1 : blocks[1],
      periodic[2] ? 2*blocks[2]+1 : blocks[2]);
  py::dict context;
  context["kind"] = "box";
  context["bounds"] = bounds;
  context["blocks"] = blocks;
  context["periodic"] = periodic;
  context["init_mem"] = init_mem;
  context["power"] = radii != nullptr;
  add_grid_context(*original, *observer, context);
  return collect<Ordinary, Observer, voro::c_loop_all>(
      *original, *observer, compute, points, ids, radii, periodic, nullptr, context);
}

template <class Ordinary, class Observer>
py::dict observe_periodic(const Doubles& points, const Integers& ids,
                          const std::array<double, 6>& params,
                          const std::array<int, 3>& blocks, int init_mem,
                          const Doubles* radii) {
  auto make = [&](auto type) {
    using Container = typename decltype(type)::type;
    return std::make_unique<Container>(params[0], params[1], params[2],
        params[3], params[4], params[5], blocks[0], blocks[1], blocks[2], init_mem);
  };
  auto original = make(std::common_type<Ordinary>{});
  auto observer = make(std::common_type<Observer>{});
  voro::witness_unitcell seed(params[0], params[1], params[2],
                             params[3], params[4], params[5]);
  require_same_geometry(original->unit_voro, seed.unit_voro);
  require_same_geometry(observer->unit_voro, seed.unit_voro);
  if (!same_bits(original->unit_voro.tol, seed.unit_voro.tol) ||
      !same_bits(original->unit_voro.tol_cu, seed.unit_voro.tol_cu) ||
      !same_bits(original->unit_voro.big_tol, seed.unit_voro.big_tol))
    throw std::runtime_error("native witness noninterference: seed tolerance");
  voro::witness_compute<Observer> compute(*observer, 2*blocks[0]+1,
                                          2*observer->ey+1, 2*observer->ez+1);
  py::dict context;
  context["kind"] = "periodic";
  context["cell_params"] = params;
  context["blocks"] = blocks;
  context["periodic"] = std::array<bool, 3>{true,true,true};
  context["init_mem"] = init_mem;
  context["power"] = radii != nullptr;
  py::dict seed_tolerances;
  seed_tolerances["tol"] = original->unit_voro.tol;
  seed_tolerances["tol_cu"] = original->unit_voro.tol_cu;
  seed_tolerances["big_tol"] = original->unit_voro.big_tol;
  context["seed_tolerances"] = seed_tolerances;
  add_grid_context(*original, *observer, context);
  return collect<Ordinary, Observer, voro::c_loop_all_periodic>(
      *original, *observer, compute, points, ids, radii, {true,true,true},
      &seed.unit_voro, context);
}

constexpr std::size_t ghost_source_limit = 1'000'000;
constexpr std::size_t ghost_occurrence_limit = 262'144;
constexpr std::size_t ghost_observer_bytes = 64u * 1024u * 1024u;

struct GhostBatchBudget {
  // Conservative charged CPython packet envelope, independently calibrated
  // against recursive getsizeof on representative selected packets. This is
  // a resource policy, not a promise of exact allocator RSS. Current native
  // observer allocation is added as peak; earlier observers are destroyed.
  static constexpr std::size_t site_bytes = 2048;
  static constexpr std::size_t packet_bytes = 16384;
  static constexpr std::size_t origin_bytes = 2048;
  static constexpr std::size_t face_bytes = 2048;
  static constexpr std::size_t vertex_bytes = 512;
  static constexpr std::size_t edge_bytes = 512;
  std::size_t retained = 0;
  std::size_t origins = 0;
  std::size_t faces = 0;

  void preflight(std::size_t sites_per_query, std::size_t queries) {
    const std::size_t sites = pre::checked_multiply(sites_per_query, queries,
                                                     "ghost batch site rows");
    retained = pre::checked_add(
        pre::checked_multiply(sites, site_bytes, "ghost batch sites"),
        pre::checked_multiply(queries, packet_bytes, "ghost batch packets"),
        "ghost batch payload");
    if (retained > ghost_observer_bytes)
      throw std::runtime_error("GHOST_CERTIFICATION_RESOURCE: retained batch base");
  }

  void charge(std::size_t new_origins, std::size_t new_faces,
              std::size_t new_vertices, std::size_t new_edges,
              std::size_t current_native_bytes) {
    try {
    const std::size_t all_origins = pre::checked_add(
        origins, new_origins, "ghost batch source tokens");
    const std::size_t all_faces = pre::checked_add(
        faces, new_faces, "ghost batch occurrences");
    if (all_origins > ghost_source_limit || all_faces > ghost_occurrence_limit)
      throw std::runtime_error("GHOST_CERTIFICATION_RESOURCE: batch token/occurrence limit");
    const std::size_t topology = pre::checked_add(
        pre::checked_multiply(new_vertices, vertex_bytes, "ghost batch vertices"),
        pre::checked_multiply(new_edges, edge_bytes, "ghost batch edges"),
        "ghost batch topology payload");
    const std::size_t dynamic = pre::checked_add(
        pre::checked_multiply(new_origins, origin_bytes, "ghost batch origins"),
        pre::checked_multiply(new_faces, face_bytes, "ghost batch faces"),
        "ghost batch source/face payload");
    const std::size_t next = pre::checked_add(retained,
        pre::checked_add(dynamic, topology, "ghost batch dynamic payload"),
        "ghost batch retained payload");
    if (pre::checked_add(next, current_native_bytes,
                         "ghost batch peak observer memory") > ghost_observer_bytes)
      throw std::runtime_error("GHOST_CERTIFICATION_RESOURCE: batch retained/observer memory");
    origins = all_origins;
    faces = all_faces;
    retained = next;
    } catch (const py::value_error& error) {
      throw std::runtime_error(std::string("GHOST_CERTIFICATION_RESOURCE: ") +
                               error.what());
    }
  }
};

void require_ghost_eager_box(const std::array<int,3>& blocks,
                             const std::array<bool,3>& periodic,
                             int init_mem, bool power) {
  pre::ByteEstimate estimate;
  const int block_count = pre::checked_int_multiply(
      pre::checked_int_multiply(blocks[0], blocks[1], "ghost box blocks"),
      blocks[2], "ghost box blocks");
  estimate.add_slots(static_cast<std::size_t>(block_count),
      2*sizeof(void*) + 2*sizeof(int), "ghost observer block tables");
  estimate.add_slots(static_cast<std::size_t>(block_count)*init_mem,
      sizeof(int) + (power ? 4u : 3u)*sizeof(double),
      "ghost observer eager primary storage");
  int mask_count = 1;
  for (int axis = 0; axis < 3; ++axis)
    mask_count = pre::checked_int_multiply(mask_count,
        periodic[axis] ? pre::checked_int_add(
            pre::checked_int_multiply(2, blocks[axis], "ghost mask"),
            1, "ghost mask") : blocks[axis], "ghost mask");
  estimate.add_slots(static_cast<std::size_t>(mask_count),
                     sizeof(unsigned int), "ghost observer mask");
  if (estimate.total() > ghost_observer_bytes)
    throw std::runtime_error("GHOST_CERTIFICATION_RESOURCE: eager observer memory");
}

void require_ghost_eager_periodic(const std::array<double,6>& params,
                                  const std::array<int,3>& blocks,
                                  int init_mem, bool power) {
  const auto estimate = pre::estimate_periodic_3d_resources(
      params, blocks, init_mem, power ? 4 : 3);
  if (estimate.known_eager_allocation.total() > ghost_observer_bytes)
    throw std::runtime_error("GHOST_CERTIFICATION_RESOURCE: eager observer memory");
}

[[noreturn]] void selected_ghost_failure(int query_index,
                                         const std::exception& error) {
  const std::string message = error.what();
  const bool insertion = message.find("GHOST_BACKEND_INSERTION:") == 0;
  const bool resource = dynamic_cast<const std::bad_alloc*>(&error) != nullptr ||
      message.find("GHOST_CERTIFICATION_RESOURCE:") == 0;
  const bool provenance = message.find("native witness unknown provenance") == 0 ||
      message.find("native witness mixed face provenance") == 0 ||
      message.find("native witness surviving construction") == 0 ||
      message.find("GHOST_PROVENANCE_INCONSISTENT:") == 0;
  const std::string code = insertion ? "GHOST_BACKEND_INSERTION" :
      resource ? "GHOST_CERTIFICATION_RESOURCE" :
      provenance ? "GHOST_PROVENANCE_INCONSISTENT" : "GHOST_NATIVE_UNSUPPORTED";
  const std::string stage = insertion ? "insertion" :
      (resource || provenance) ? "provenance" : "native";
  throw py::value_error("ghost_native:" + stage + ":" +
      std::to_string(query_index) + ":" + code + ":" +
      message.substr(0, 240));
}

template <class Ordinary, class Observer, class AllLoop, class OrderLoop>
py::dict collect_selected_ghost(Ordinary& ordinary, Observer& observer,
                                voro::witness_compute<Observer>& compute,
                                const Doubles& points, const Integers& ids,
                                const Doubles* radii,
                                const std::array<double, 3>& query,
                                const py::object& ghost_radius,
                                const std::array<bool, 3>& periodic,
                                ObservedCell* seed, py::dict context,
                                int query_index, GhostBatchBudget& batch_budget) {
  const int n = pre::checked_int(points.shape(0), "ghost internal ID");
  pre::checked_int_add(n, 1, "augmented ghost count");
  if (static_cast<std::size_t>(n) >= ghost_source_limit)
    throw std::runtime_error("GHOST_CERTIFICATION_RESOURCE: native source limit");
  // This is a conservative preflight for the additional matched observing
  // population. Dynamic origin and serialized-topology accounting follows.
  pre::ByteEstimate observer_estimate;
  observer_estimate.add_slots(static_cast<std::size_t>(n) + 1,
      sizeof(int) + (radii ? 4u : 3u) * sizeof(double),
      "ghost observer population");
  if (observer_estimate.total() > ghost_observer_bytes)
    throw std::runtime_error("GHOST_CERTIFICATION_RESOURCE: observer memory limit");

  auto coordinates = points.unchecked<2>();
  auto identifiers = ids.unchecked<1>();
  const double* native_radii = radii ? radii->data() : nullptr;
  for (int i = 0; i < n; ++i) {
    if constexpr (std::is_base_of_v<voro::radius_poly, Ordinary>) {
      ordinary.put(identifiers(i), coordinates(i,0), coordinates(i,1),
                   coordinates(i,2), native_radii[i]);
      observer.put(identifiers(i), coordinates(i,0), coordinates(i,1),
                   coordinates(i,2), native_radii[i]);
    } else {
      ordinary.put(identifiers(i), coordinates(i,0), coordinates(i,1),
                   coordinates(i,2));
      observer.put(identifiers(i), coordinates(i,0), coordinates(i,1),
                   coordinates(i,2));
    }
  }
  voro::particle_order ordinary_order(2), observer_order(2);
  if constexpr (std::is_base_of_v<voro::radius_poly, Ordinary>) {
    const double r = py::cast<double>(ghost_radius);
    ordinary.put(ordinary_order, n, query[0], query[1], query[2], r);
    observer.put(observer_order, n, query[0], query[1], query[2], r);
  } else {
    ordinary.put(ordinary_order, n, query[0], query[1], query[2]);
    observer.put(observer_order, n, query[0], query[1], query[2]);
  }

  AllLoop original_all(ordinary), observed_all(observer);
  std::vector<py::object> stored_sites(static_cast<std::size_t>(n + 1));
  std::vector<unsigned char> seen(stored_sites.size(), 0);
  bool more = original_all.start(), observed_more = observed_all.start();
  while (more || observed_more) {
    if (more != observed_more)
      throw std::runtime_error("GHOST_BACKEND_INSERTION: observer population count");
    int owner, other_owner;
    double x, y, z, r, ox, oy, oz, other_radius;
    original_all.pos(owner, x, y, z, r);
    observed_all.pos(other_owner, ox, oy, oz, other_radius);
    if (owner != other_owner || !same_bits(x, ox) || !same_bits(y, oy) ||
        !same_bits(z, oz) || !same_bits(r, other_radius) ||
        original_all.i != observed_all.i ||
        original_all.j != observed_all.j ||
        original_all.k != observed_all.k ||
        original_all.ijk != observed_all.ijk ||
        original_all.q != observed_all.q)
      throw std::runtime_error("GHOST_BACKEND_INSERTION: observer storage mismatch");
    if (owner < 0 || owner > n || seen[owner]++)
      throw std::runtime_error("GHOST_BACKEND_INSERTION: invalid augmented identity");
    py::object radius = radii ? py::object(py::float_(r)) : py::object(py::none());
    py::dict stored;
    stored["id"] = owner;
    stored["site"] = std::array<double,3>{x,y,z};
    stored["radius"] = radius;
    stored["block"] = std::array<int,3>{original_all.i, original_all.j,
                                          original_all.k};
    stored["block_index"] = original_all.ijk;
    stored["block_slot"] = original_all.q;
    stored_sites[owner] = stored;
    more = original_all.inc();
    observed_more = observed_all.inc();
  }
  py::list sites;
  for (int owner = 0; owner <= n; ++owner) {
    if (!seen[owner])
      throw std::runtime_error("GHOST_BACKEND_INSERTION: missing augmented site");
    sites.append(stored_sites[owner]);
  }

  OrderLoop original_selected(ordinary, ordinary_order);
  OrderLoop observed_selected(observer, observer_order);
  if (!original_selected.start() || !observed_selected.start())
    throw std::runtime_error("GHOST_BACKEND_INSERTION: selected slot missing");
  int owner, observed_owner;
  double x, y, z, r, ox, oy, oz, other_radius;
  original_selected.pos(owner, x, y, z, r);
  observed_selected.pos(observed_owner, ox, oy, oz, other_radius);
  if (owner != n || observed_owner != n ||
      !same_bits(x, ox) || !same_bits(y, oy) || !same_bits(z, oz) ||
      !same_bits(r, other_radius) ||
      original_selected.ijk != observed_selected.ijk ||
      original_selected.q != observed_selected.q ||
      original_selected.ijk != py::cast<int>(
          py::cast<py::dict>(stored_sites[n])["block_index"]) ||
      original_selected.q != py::cast<int>(
          py::cast<py::dict>(stored_sites[n])["block_slot"]) ||
      original_selected.inc() || observed_selected.inc())
    throw std::runtime_error("GHOST_BACKEND_INSERTION: selected slot mismatch");

  voro::voronoicell_neighbor original_cell;
  ObservedCell observed_cell;
  observed_cell.limit_ghost_origins(ghost_occurrence_limit);
  if (seed) observed_cell.begin_periodic(n, *seed);
  else observed_cell.begin_box(n, periodic);
  const bool original_success = ordinary.compute_cell(original_cell,
                                                      original_selected);
  const bool observed_success = compute.compute_cell(
      observed_cell, observed_selected.ijk, observed_selected.q,
      observed_selected.i, observed_selected.j, observed_selected.k);
  if (original_success != observed_success)
    throw std::runtime_error("GHOST_NATIVE_UNSUPPORTED: selected computation parity");
  const int block_count = [&] {
    if constexpr (std::is_base_of_v<voro::container_periodic_base, Observer>)
      return observer.oxyz;
    else return observer.nxyz;
  }();
  pre::ByteEstimate observed_allocation;
  observed_allocation.add_slots(static_cast<std::size_t>(block_count),
      2*sizeof(void*) + 2*sizeof(int) +
      (seed ? sizeof(char) : 0u), "ghost observer block tables");
  for (int block = 0; block < block_count; ++block)
    if (observer.mem[block] > 0)
      observed_allocation.add_slots(static_cast<std::size_t>(observer.mem[block]),
          sizeof(int) + (radii ? 4u : 3u)*sizeof(double),
          "ghost observer particle capacity");
  observed_allocation.add_slots(observed_cell.origins.capacity(),
                                 sizeof(Origin), "ghost source tokens");
  observed_allocation.add_slots(static_cast<std::size_t>(observed_cell.current_vertices),
      4*sizeof(double) + 16*sizeof(int), "ghost observer topology envelope");
  if (observed_allocation.total() > ghost_observer_bytes)
    throw std::runtime_error("GHOST_CERTIFICATION_RESOURCE: observer memory limit");
  const int face_count = original_success ? observed_cell.number_of_faces() : 0;
  if (face_count < 0)
    throw std::runtime_error("GHOST_PROVENANCE_INCONSISTENT: negative face count");
  std::size_t edge_count = 0;
  if (original_success)
    for (int vertex = 0; vertex < observed_cell.p; ++vertex)
      try {
        edge_count = pre::checked_add(edge_count,
            static_cast<std::size_t>(observed_cell.nu[vertex]),
            "ghost batch edge incidence");
      } catch (const py::value_error& error) {
        throw std::runtime_error(std::string("GHOST_CERTIFICATION_RESOURCE: ") +
                                 error.what());
      }
  batch_budget.charge(observed_cell.origins.size(),
                      static_cast<std::size_t>(face_count),
                      original_success ? static_cast<std::size_t>(observed_cell.p) : 0,
                      edge_count, observed_allocation.total());
  py::dict selected_cell = cell_dict(original_cell, observed_cell,
      original_success, n, {x,y,z}, ghost_radius);
  if (selected_cell["faces"].cast<py::list>().size() !=
      static_cast<std::size_t>(face_count))
    throw std::runtime_error("GHOST_PROVENANCE_INCONSISTENT: face count changed");
  py::dict packet;
  packet["schema"] = "wp7-selected-ghost-3d-v1";
  packet["query_index"] = query_index;
  packet["ghost_internal_id"] = n;
  packet["sites"] = sites;
  py::list cells;
  cells.append(selected_cell);
  packet["cells"] = cells;
  packet["context"] = context;
  py::dict build = build_metadata();
  ghost_metadata(build);
  packet["build"] = build;
  return packet;
}

template <class Ordinary, class Observer>
py::list observe_selected_box(const Doubles& points, const Integers& ids,
                              const Bounds& bounds, const std::array<int,3>& blocks,
                              const std::array<bool,3>& periodic, int init_mem,
                              const Doubles& queries, const Doubles* radii,
                              const Doubles* ghost_radii) {
  py::list packets;
  GhostBatchBudget budget;
  if (queries.shape(0) != 0)
    try {
      require_ghost_eager_box(blocks, periodic, init_mem, radii != nullptr);
      budget.preflight(static_cast<std::size_t>(points.shape(0)) + 1,
                       static_cast<std::size_t>(queries.shape(0)));
    } catch (const std::exception& error) {
      throw py::value_error(std::string("ghost_native:preparation:None:") +
          "GHOST_CERTIFICATION_RESOURCE:" + error.what());
    }
  auto q = queries.unchecked<2>();
  const double* ghost_r = ghost_radii ? ghost_radii->data() : nullptr;
  for (py::ssize_t index = 0; index < queries.shape(0); ++index) {
    try {
    auto make = [&](auto type) {
      using C = typename decltype(type)::type;
      return std::make_unique<C>(bounds[0][0], bounds[0][1], bounds[1][0],
          bounds[1][1], bounds[2][0], bounds[2][1], blocks[0], blocks[1],
          blocks[2], periodic[0], periodic[1], periodic[2], init_mem);
    };
    auto original = make(std::common_type<Ordinary>{});
    auto observer = make(std::common_type<Observer>{});
    voro::witness_compute<Observer> compute(*observer,
        periodic[0] ? 2*blocks[0]+1 : blocks[0],
        periodic[1] ? 2*blocks[1]+1 : blocks[1],
        periodic[2] ? 2*blocks[2]+1 : blocks[2]);
    py::dict context;
    context["kind"] = "box";
    context["bounds"] = bounds;
    context["blocks"] = blocks;
    context["periodic"] = periodic;
    context["init_mem"] = init_mem;
    context["power"] = radii != nullptr;
    add_grid_context(*original, *observer, context);
    py::object gr = ghost_r ? py::object(py::float_(ghost_r[index])) : py::none();
    packets.append(collect_selected_ghost<Ordinary, Observer,
        voro::c_loop_all, voro::c_loop_order>(*original, *observer, compute,
        points, ids, radii, {q(index,0),q(index,1),q(index,2)}, gr,
        periodic, nullptr, context, static_cast<int>(index), budget));
    } catch (const std::exception& error) {
      selected_ghost_failure(static_cast<int>(index), error);
    }
  }
  return packets;
}

template <class Ordinary, class Observer>
py::list observe_selected_periodic(const Doubles& points, const Integers& ids,
                                   const std::array<double,6>& params,
                                   const std::array<int,3>& blocks, int init_mem,
                                   const Doubles& queries, const Doubles* radii,
                                   const Doubles* ghost_radii) {
  py::list packets;
  GhostBatchBudget budget;
  if (queries.shape(0) != 0)
    try {
      require_ghost_eager_periodic(params, blocks, init_mem, radii != nullptr);
      budget.preflight(static_cast<std::size_t>(points.shape(0)) + 1,
                       static_cast<std::size_t>(queries.shape(0)));
    } catch (const std::exception& error) {
      throw py::value_error(std::string("ghost_native:preparation:None:") +
          "GHOST_CERTIFICATION_RESOURCE:" + error.what());
    }
  auto q = queries.unchecked<2>();
  const double* ghost_r = ghost_radii ? ghost_radii->data() : nullptr;
  for (py::ssize_t index = 0; index < queries.shape(0); ++index) {
    try {
    auto make = [&](auto type) {
      using C = typename decltype(type)::type;
      return std::make_unique<C>(params[0], params[1], params[2], params[3],
          params[4], params[5], blocks[0], blocks[1], blocks[2], init_mem);
    };
    auto original = make(std::common_type<Ordinary>{});
    auto observer = make(std::common_type<Observer>{});
    voro::witness_unitcell seed(params[0], params[1], params[2], params[3],
                                params[4], params[5]);
    require_same_geometry(original->unit_voro, seed.unit_voro);
    require_same_geometry(observer->unit_voro, seed.unit_voro);
    if (!same_bits(original->unit_voro.tol, seed.unit_voro.tol) ||
        !same_bits(original->unit_voro.tol_cu, seed.unit_voro.tol_cu) ||
        !same_bits(original->unit_voro.big_tol, seed.unit_voro.big_tol))
      throw std::runtime_error("GHOST_NATIVE_UNSUPPORTED: seed tolerance parity");
    voro::witness_compute<Observer> compute(*observer, 2*blocks[0]+1,
                                            2*observer->ey+1, 2*observer->ez+1);
    py::dict context;
    context["kind"] = "periodic";
    context["cell_params"] = params;
    context["blocks"] = blocks;
    context["periodic"] = std::array<bool,3>{true,true,true};
    context["init_mem"] = init_mem;
    context["power"] = radii != nullptr;
    py::dict tolerances;
    tolerances["tol"] = original->unit_voro.tol;
    tolerances["tol_cu"] = original->unit_voro.tol_cu;
    tolerances["big_tol"] = original->unit_voro.big_tol;
    context["seed_tolerances"] = tolerances;
    add_grid_context(*original, *observer, context);
    py::object gr = ghost_r ? py::object(py::float_(ghost_r[index])) : py::none();
    packets.append(collect_selected_ghost<Ordinary, Observer,
        voro::c_loop_all_periodic, voro::c_loop_order_periodic>(
        *original, *observer, compute, points, ids, radii,
        {q(index,0),q(index,1),q(index,2)}, gr,
        {true,true,true}, &seed.unit_voro, context, static_cast<int>(index),
        budget));
    } catch (const std::exception& error) {
      selected_ghost_failure(static_cast<int>(index), error);
    }
  }
  return packets;
}

// Exercise the two actual vendored weighted primitives independently; their
// different evaluation trees are part of the source arithmetic, not a rewrite.
class RadiusPrimitiveProbe : public voro::radius_poly {
 public:
  py::dict evaluate(double distance_squared, double owner_radius,
                    double neighbor_radius, double maximum_squared_radius) {
    double storage[8] = {0,0,0,owner_radius,0,0,0,neighbor_radius};
    double* blocks[] = {storage};
    ppr = blocks;
    max_radius = std::max(owner_radius, neighbor_radius);
    r_init(0,0);
    const double ordinary = r_scale(distance_squared,0,1);
    double checked = distance_squared;
    const bool passed = r_scale_check(checked, maximum_squared_radius,0,1);
    py::dict out;
    out["r_scale"] = ordinary;
    out["r_scale_check_offset"] = checked;
    out["r_scale_check_passes"] = passed;
    return out;
  }
};
}  // namespace

void register_bindings(py::module_& module) {
  // Consistency data, never qualification authority. Inspection contains only
  // integer/string/bool materialization and is safe under hostile FP controls.
  pyvoro2::native_runtime::inspection_def(module, "_spatial_witness_profile", [] {
    py::dict build = profile_metadata();
    ghost_metadata(build);
    return build;
  });
  pyvoro2::native_runtime::guarded_def(module, "_observe_ghost_box", [](Doubles points, Integers ids, Bounds bounds,
      std::array<int, 3> blocks, std::array<bool, 3> periodic, int init_mem,
      Doubles queries, py::object radii_object, py::object ghost_radii_object) {
    if (queries.shape(0) != 0)
      try {
        require_environment();
      } catch (const std::exception& error) {
        throw py::value_error(std::string("ghost_native:native:None:") +
            "GHOST_NATIVE_UNSUPPORTED:" + error.what());
      }
    Doubles radii, ghost_radii;
    const Doubles* native_radii = nullptr;
    const Doubles* native_ghost_radii = nullptr;
    if (!radii_object.is_none()) {
      radii = native_runtime::guarded_cast<Doubles>(radii_object);
      native_radii = &radii;
    }
    if (!ghost_radii_object.is_none()) {
      ghost_radii = native_runtime::guarded_cast<Doubles>(ghost_radii_object);
      native_ghost_radii = &ghost_radii;
    }
    if ((native_radii == nullptr) != (native_ghost_radii == nullptr))
      throw py::value_error("ghost radii must accompany persistent radii");
    pre::preflight_box<3>(points, ids, native_radii, bounds, blocks,
                          periodic, init_mem, native_radii ? 4 : 3,
                          &queries, native_ghost_radii, true, false);
    pre::checked_int_add(pre::checked_int(points.shape(0), "ghost internal ID"),
                         1, "augmented ghost count");
    if (queries.shape(0) == 0) return py::list();
    if (native_radii) return observe_selected_box<voro::container_poly,
        voro::witness_container_poly>(points, ids, bounds, blocks, periodic,
                                     init_mem, queries, native_radii,
                                     native_ghost_radii);
    return observe_selected_box<voro::container, voro::witness_container>(
        points, ids, bounds, blocks, periodic, init_mem, queries, nullptr,
        nullptr);
  }, py::arg("points"), py::arg("ids"), py::arg("bounds"),
     py::arg("blocks"), py::arg("periodic"), py::arg("init_mem"),
     py::arg("queries"), py::arg("radii") = py::none(),
     py::arg("ghost_radii") = py::none());

  pyvoro2::native_runtime::guarded_def(module, "_observe_ghost_periodic", [](Doubles points, Integers ids,
      std::array<double, 6> params, std::array<int, 3> blocks, int init_mem,
      Doubles queries, py::object radii_object, py::object ghost_radii_object) {
    if (queries.shape(0) != 0)
      try {
        require_environment();
      } catch (const std::exception& error) {
        throw py::value_error(std::string("ghost_native:native:None:") +
            "GHOST_NATIVE_UNSUPPORTED:" + error.what());
      }
    Doubles radii, ghost_radii;
    const Doubles* native_radii = nullptr;
    const Doubles* native_ghost_radii = nullptr;
    if (!radii_object.is_none()) {
      radii = native_runtime::guarded_cast<Doubles>(radii_object);
      native_radii = &radii;
    }
    if (!ghost_radii_object.is_none()) {
      ghost_radii = native_runtime::guarded_cast<Doubles>(ghost_radii_object);
      native_ghost_radii = &ghost_radii;
    }
    if ((native_radii == nullptr) != (native_ghost_radii == nullptr))
      throw py::value_error("ghost radii must accompany persistent radii");
    pre::preflight_periodic_3d(points, ids, native_radii, params, blocks,
                               init_mem, native_radii ? 4 : 3, &queries,
                               native_ghost_radii, true, false);
    pre::checked_int_add(pre::checked_int(points.shape(0), "ghost internal ID"),
                         1, "augmented ghost count");
    if (queries.shape(0) == 0) return py::list();
    if (native_radii) return observe_selected_periodic<
        voro::container_periodic_poly, voro::witness_container_periodic_poly>(
        points, ids, params, blocks, init_mem, queries, native_radii,
        native_ghost_radii);
    return observe_selected_periodic<voro::container_periodic,
        voro::witness_container_periodic>(points, ids, params, blocks,
                                          init_mem, queries, nullptr, nullptr);
  }, py::arg("points"), py::arg("ids"), py::arg("cell_params"),
     py::arg("blocks"), py::arg("init_mem"), py::arg("queries"),
     py::arg("radii") = py::none(), py::arg("ghost_radii") = py::none());

  pyvoro2::native_runtime::guarded_def(module, "_observe_box", [](Doubles points, Integers ids, Bounds bounds,
      std::array<int, 3> blocks, std::array<bool, 3> periodic, int init_mem,
      py::object radii_object) {
    require_environment();
    Doubles radii;
    const Doubles* native_radii = nullptr;
    if (!radii_object.is_none()) {
      radii = native_runtime::guarded_cast<Doubles>(radii_object);
      native_radii = &radii;
    }
    pre::preflight_box<3>(points, ids, native_radii, bounds, blocks, periodic,
                          init_mem, native_radii ? 4 : 3);
    if (native_radii) return observe_box<voro::container_poly,
        voro::witness_container_poly>(points, ids, bounds, blocks, periodic,
                                     init_mem, native_radii);
    return observe_box<voro::container, voro::witness_container>(
        points, ids, bounds, blocks, periodic, init_mem, nullptr);
  }, py::arg("points"), py::arg("ids"), py::arg("bounds"), py::arg("blocks"),
     py::arg("periodic"), py::arg("init_mem"), py::arg("radii") = py::none());

  pyvoro2::native_runtime::guarded_def(module, "_observe_periodic", [](Doubles points, Integers ids,
      std::array<double, 6> params, std::array<int, 3> blocks, int init_mem,
      py::object radii_object) {
    require_environment();
    Doubles radii;
    const Doubles* native_radii = nullptr;
    if (!radii_object.is_none()) {
      radii = native_runtime::guarded_cast<Doubles>(radii_object);
      native_radii = &radii;
    }
    pre::preflight_periodic_3d(points, ids, native_radii, params, blocks, init_mem,
                              native_radii ? 4 : 3);
    if (native_radii) return observe_periodic<voro::container_periodic_poly,
        voro::witness_container_periodic_poly>(points, ids, params, blocks,
                                              init_mem, native_radii);
    return observe_periodic<voro::container_periodic,
        voro::witness_container_periodic>(points, ids, params, blocks, init_mem,
                                         nullptr);
  }, py::arg("points"), py::arg("ids"), py::arg("cell_params"),
     py::arg("blocks"), py::arg("init_mem"), py::arg("radii") = py::none());

  pyvoro2::native_runtime::guarded_def(module, "_test_power_offset_order", [](double distance_squared,
      double owner_radius, double neighbor_radius, double max_radius_squared) {
    require_environment();
    return RadiusPrimitiveProbe{}.evaluate(distance_squared, owner_radius,
                                            neighbor_radius, max_radius_squared);
  }, py::arg("distance_squared"), py::arg("owner_radius"),
     py::arg("neighbor_radius"), py::arg("max_radius_squared") = 1e100);
}
}  // namespace pyvoro2::native_witness
