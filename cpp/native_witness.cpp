#include "native_witness.hpp"
#include "native_preconditions.hpp"

#include <pybind11/stl.h>

#include <cfloat>
#include <memory>
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
  volatile double minimum = std::numeric_limits<double>::min();
  volatile double two = 2.0;
  volatile double half = minimum / two;
  if (!(half > 0.0 && half < minimum))
    throw std::runtime_error("native witness requires gradual binary64 underflow");
}

py::dict build_metadata() {
  py::dict build;
  build["binary64"] = true;
  build["round_to_nearest"] = true;
  build["gradual_underflow"] = true;
  build["float_eval_method"] = FLT_EVAL_METHOD;
  build["fast_math"] = false;
  build["fp_contract"] = "off";
  build["fp_contract_scope"] =
      "all _core translation units, including ordinary producer and clipping";
  build["native_fp_policy"] = "binary64-noncontracting-v1";
  build["ipo"] = false;
  build["source_coupled_seed_replay"] = true;
  build["source_coupled_compute"] = true;
#ifdef __VERSION__
  build["compiler"] = __VERSION__;
#else
  build["compiler"] = "MSVC " PYVORO2_WITNESS_COMPILER_VERSION;
#endif
  build["source_sha256"] = PYVORO2_WITNESS_SOURCE_SHA256;
  build["source_sha256_scope"] = "listed files; not a complete source fingerprint";
  build["tolerance"] = voro::tolerance;
  build["big_tolerance_factor"] = voro::big_tolerance_fac;
  build["max_unit_voro_shells"] = voro::max_unit_voro_shells;
  return build;
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
    if (owner < 0 || static_cast<std::size_t>(owner) >= seen.size() || seen[owner])
      throw std::runtime_error("native witness invalid persistent site identity");
    seen[owner] = 1;
    py::object radius = radii ? py::object(py::float_(r)) : py::object(py::none());
    py::dict stored;
    stored["id"] = owner;
    stored["site"] = std::array<double, 3>{x,y,z};
    stored["radius"] = radius;
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
  return collect<Ordinary, Observer, voro::c_loop_all_periodic>(
      *original, *observer, compute, points, ids, radii, {true,true,true},
      &seed.unit_voro, context);
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
  module.def("_observe_box", [](Doubles points, Integers ids, Bounds bounds,
      std::array<int, 3> blocks, std::array<bool, 3> periodic, int init_mem,
      py::object radii_object) {
    require_environment();
    Doubles radii;
    const Doubles* native_radii = nullptr;
    if (!radii_object.is_none()) {
      radii = py::cast<Doubles>(radii_object);
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

  module.def("_observe_periodic", [](Doubles points, Integers ids,
      std::array<double, 6> params, std::array<int, 3> blocks, int init_mem,
      py::object radii_object) {
    require_environment();
    Doubles radii;
    const Doubles* native_radii = nullptr;
    if (!radii_object.is_none()) {
      radii = py::cast<Doubles>(radii_object);
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

  module.def("_test_power_offset_order", [](double distance_squared,
      double owner_radius, double neighbor_radius, double max_radius_squared) {
    require_environment();
    return RadiusPrimitiveProbe{}.evaluate(distance_squared, owner_radius,
                                            neighbor_radius, max_radius_squared);
  }, py::arg("distance_squared"), py::arg("owner_radius"),
     py::arg("neighbor_radius"), py::arg("max_radius_squared") = 1e100);
}
}  // namespace pyvoro2::native_witness
