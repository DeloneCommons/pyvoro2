#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "voro++.hh"
#include "native_preconditions.hpp"
#include "locate_source.hpp"
#include "native_witness.hpp"

namespace py = pybind11;
using namespace voro;
namespace native = pyvoro2::native_preconditions;

namespace {

struct OutputOpts {
  bool vertices;
  bool adjacency;
  bool faces;
};

[[noreturn]] void ghost_bridge_failure(int query_index,
                                       const std::exception& error) {
  const std::string message = error.what();
  const bool insertion = message.find("GHOST_BACKEND_INSERTION:") == 0;
  const bool resource = dynamic_cast<const std::bad_alloc*>(&error) != nullptr ||
      message.find("GHOST_CERTIFICATION_RESOURCE:") == 0;
  const std::string code = insertion ? "GHOST_BACKEND_INSERTION" :
      resource ? "GHOST_CERTIFICATION_RESOURCE" : "GHOST_NATIVE_UNSUPPORTED";
  const std::string stage = insertion ? "insertion" : resource ? "provenance" :
      "native";
  throw py::value_error("ghost_native:" + stage + ":" +
      std::to_string(query_index) + ":" + code + ":" +
      message.substr(0, 240));
}

struct ReplayedGhostInsertion {
  int block;
  std::array<double, 3> site;
};

// voro_base keeps these stepping helpers protected. Use the same expressions
// as v_base.hh to replay the locator independently of the producing put().
int replay_step_int(double a) { return a < 0 ? int(a)-1 : int(a); }
int replay_step_mod(int a, int b) { return a >= 0 ? a%b : b-1-(b-1-a)%b; }
int replay_step_div(int a, int b) { return a >= 0 ? a/b : -1+(a+1)/b; }

// Replay the unchanged source locator's arithmetic without changing the
// producing container. The check below compares every replayed block/slot,
// remapped coordinate, ID and radius against actual primary storage.
template <class ContainerT>
ReplayedGhostInsertion replay_ghost_insertion(const ContainerT& con,
                                               std::array<double, 3> site) {
  double &x = site[0], &y = site[1], &z = site[2];
  if constexpr (std::is_base_of_v<voro::container_periodic_base, ContainerT>) {
    int k = replay_step_int(z * con.zsp);
    if (k < 0 || k >= con.nz) {
      const int ak = replay_step_div(k, con.nz);
      z -= ak * con.bz; y -= ak * con.byz; x -= ak * con.bxz;
      k -= ak * con.nz;
    }
    int j = replay_step_int(y * con.ysp);
    if (j < 0 || j >= con.ny) {
      const int aj = replay_step_div(j, con.ny);
      y -= aj * con.by; x -= aj * con.bxy; j -= aj * con.ny;
    }
    int i = replay_step_int(x * con.xsp);
    if (i < 0 || i >= con.nx) {
      const int ai = replay_step_div(i, con.nx);
      x -= ai * con.bx; i -= ai * con.nx;
    }
    j += con.ey; k += con.ez;
    return {i + con.nx * (j + con.oy * k), site};
  } else {
    int i = replay_step_int((x - con.ax) * con.xsp);
    if (con.xperiodic) {
      const int mapped = replay_step_mod(i, con.nx);
      x += con.boxx * (mapped - i); i = mapped;
    } else if (i < 0 || i >= con.nx)
      throw std::runtime_error("GHOST_BACKEND_INSERTION: x locator rejected input");
    int j = replay_step_int((y - con.ay) * con.ysp);
    if (con.yperiodic) {
      const int mapped = replay_step_mod(j, con.ny);
      y += con.boxy * (mapped - j); j = mapped;
    } else if (j < 0 || j >= con.ny)
      throw std::runtime_error("GHOST_BACKEND_INSERTION: y locator rejected input");
    int k = replay_step_int((z - con.az) * con.zsp);
    if (con.zperiodic) {
      const int mapped = replay_step_mod(k, con.nz);
      z += con.boxz * (mapped - k); k = mapped;
    } else if (k < 0 || k >= con.nz)
      throw std::runtime_error("GHOST_BACKEND_INSERTION: z locator rejected input");
    return {i + con.nx * j + con.nxy * k, site};
  }
}

OutputOpts parse_opts(const std::tuple<bool, bool, bool>& opts) {
  return OutputOpts{std::get<0>(opts), std::get<1>(opts), std::get<2>(opts)};
}

py::dict build_cell_dict(voronoicell_neighbor& cell,
                        int pid,
                        double x,
                        double y,
                        double z,
                        const OutputOpts& opts) {
  py::dict out;
  out["id"] = pid;
  out["volume"] = cell.volume();

  // Always include the generator (site) position used by Voro++ for this cell.
  // For periodic containers this is in the internal (lower-triangular)
  // coordinate system; the Python layer converts it back to Cartesian for
  // `PeriodicCell`.
  py::list site;
  site.append(x);
  site.append(y);
  site.append(z);
  out["site"] = site;

  if (opts.vertices) {
    std::vector<double> positions;
    cell.vertices(x, y, z, positions);
    py::list verts;
    for (std::size_t i = 0; i + 2 < positions.size(); i += 3) {
      py::list v;
      v.append(positions[i]);
      v.append(positions[i + 1]);
      v.append(positions[i + 2]);
      verts.append(v);
    }
    out["vertices"] = verts;
  }

  if (opts.adjacency) {
    py::list adj;
    for (int i = 0; i < cell.p; i++) {
      py::list row;
      for (int j = 0; j < cell.nu[i]; j++) row.append(cell.ed[i][j]);
      adj.append(row);
    }
    out["adjacency"] = adj;
  }

  if (opts.faces) {
    auto parse_face_vertices = [](const std::vector<int>& fflat)
        -> std::vector<std::vector<int>> {
      std::vector<std::vector<int>> face_vs;
      std::size_t k = 0;
      while (k < fflat.size()) {
        int nv = fflat[k++];
        if (nv < 0) nv = 0;
        const std::size_t nvu = static_cast<std::size_t>(nv);
        if (k + nvu > fflat.size()) {
          throw std::runtime_error("face_vertices encoding overflow");
        }
        std::vector<int> fv;
        fv.reserve(nvu);
        for (std::size_t j = 0; j < nvu; ++j) {
          fv.push_back(fflat[k++]);
        }
        face_vs.emplace_back(std::move(fv));
      }
      return face_vs;
    };

    // Voro++ face traversal methods should agree on face ordering, but we have
    // seen rare platform-dependent inconsistencies. Try both call orders.
    std::vector<int> neigh;
    std::vector<int> fflat;

    cell.face_vertices(fflat);
    cell.neighbors(neigh);
    std::vector<std::vector<int>> face_vs = parse_face_vertices(fflat);

    if (face_vs.size() != neigh.size()) {
      neigh.clear();
      fflat.clear();
      cell.neighbors(neigh);
      cell.face_vertices(fflat);
      face_vs = parse_face_vertices(fflat);
    }

    if (face_vs.size() != neigh.size()) {
      throw std::runtime_error(
          std::string("pyvoro2 internal error: mismatch between neighbors and "
                      "face_vertices counts (neighbors=") +
          std::to_string(neigh.size()) + ", faces=" +
          std::to_string(face_vs.size()) + ")");
    }

    py::list faces;
    for (std::size_t i = 0; i < face_vs.size(); ++i) {
      py::dict fd;
      fd["adjacent_cell"] = neigh[i];
      py::list fv;
      for (int vid : face_vs[i]) {
        fv.append(vid);
      }
      fd["vertices"] = fv;
      faces.append(fd);
    }

    out["faces"] = faces;
  }

  return out;
}


py::dict build_empty_ghost_dict(int query_index,
                               double x,
                               double y,
                               double z,
                               const OutputOpts& opts) {
  py::dict out;
  out["id"] = -1;
  out["empty"] = true;
  out["volume"] = 0.0;

  py::list site;
  site.append(x);
  site.append(y);
  site.append(z);
  out["site"] = site;

  out["query_index"] = query_index;

  if (opts.vertices) out["vertices"] = py::list();
  if (opts.adjacency) out["adjacency"] = py::list();
  if (opts.faces) out["faces"] = py::list();

  return out;
}

template <class ContainerT, class LoopT>
py::list compute_cells_impl(ContainerT& con, LoopT& loop, const OutputOpts& opts) {
  py::list cells;
  voronoicell_neighbor cell;

  if (loop.start())
    do {
      if (con.compute_cell(cell, loop)) {
        int pid;
        double x, y, z, r;
        loop.pos(pid, x, y, z, r);
        cells.append(build_cell_dict(cell, pid, x, y, z, opts));
      }
    } while (loop.inc());

  return cells;
}

// The ghost is one initialized member of a fresh augmented container. The
// ordering overload writes its ID before incrementing occupancy and uses the
// same native locator as the obsolete compute_ghost_cell template. The latter
// never initialized the ID and is never called from these bindings.
template <class ContainerT>
py::dict safe_selected_ghost(ContainerT& con,
                             const py::array_t<double, py::array::c_style |
                                                  py::array::forcecast>& points,
                             const py::array_t<int, py::array::c_style |
                                               py::array::forcecast>& ids,
                             const py::array_t<double, py::array::c_style |
                                                  py::array::forcecast>* radii,
                             const std::array<double, 3>& query,
                             double ghost_radius, int query_index,
                             const OutputOpts& opts) {
  const int count = native::checked_int(points.shape(0), "ghost internal ID");
  native::checked_int_add(count, 1, "augmented ghost count");
  auto p = points.unchecked<2>();
  auto id = ids.unchecked<1>();
  for (int i = 0; i < count; ++i) {
    if constexpr (std::is_base_of_v<voro::radius_poly, ContainerT>) {
      con.put(id(i), p(i, 0), p(i, 1), p(i, 2),
              radii->unchecked<1>()(i));
    } else {
      con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
    }
  }
  voro::particle_order selected_order(2);
  if constexpr (std::is_base_of_v<voro::radius_poly, ContainerT>) {
    con.put(selected_order, count, query[0], query[1], query[2], ghost_radius);
  } else {
    con.put(selected_order, count, query[0], query[1], query[2]);
  }

  const int block_count = [&] {
    if constexpr (std::is_base_of_v<voro::container_periodic_base, ContainerT>)
      return con.oxyz;
    else return con.nxyz;
  }();
  std::vector<int> expected_slots(static_cast<std::size_t>(block_count), 0);
  auto check_storage = [&](int owner, const std::array<double, 3>& supplied,
                           double supplied_radius) {
    const ReplayedGhostInsertion expected = replay_ghost_insertion(con, supplied);
    const int block = expected.block;
    if (block < 0 || block >= block_count)
      throw std::runtime_error("GHOST_BACKEND_INSERTION: replayed block outside grid");
    int& slot = expected_slots[block];
    if (slot < 0 || slot >= con.co[block] || con.id[block][slot] != owner)
      throw std::runtime_error("GHOST_BACKEND_INSERTION: replayed ID/block/slot differs");
    const double* actual = con.p[block] + con.ps * slot;
    for (int axis = 0; axis < 3; ++axis)
      if (!pyvoro2::native_witness::same_bits(actual[axis], expected.site[axis]))
        throw std::runtime_error("GHOST_BACKEND_INSERTION: stored coordinate differs");
    if constexpr (std::is_base_of_v<voro::radius_poly, ContainerT>)
      if (!pyvoro2::native_witness::same_bits(actual[3], supplied_radius))
        throw std::runtime_error("GHOST_BACKEND_INSERTION: stored radius differs");
    slot = native::checked_int_add(slot, 1, "ghost insertion slot");
  };
  for (int i = 0; i < count; ++i)
    check_storage(id(i), {p(i, 0), p(i, 1), p(i, 2)},
                  radii ? radii->unchecked<1>()(i) : 0.0);
  check_storage(count, query, ghost_radius);
  for (int block = 0; block < block_count; ++block)
    if (expected_slots[block] != con.co[block])
      throw std::runtime_error("GHOST_BACKEND_INSERTION: unaccounted native storage");

  // Count actual primary storage, not attempts at put(). In particular a
  // rectangular put can silently omit an out-of-block particle.
  std::vector<unsigned char> seen(static_cast<std::size_t>(count + 1), 0);
  int stored = 0;
  auto verify = [&](auto& loop) {
    if (loop.start()) do {
      int owner;
      double x, y, z, r;
      loop.pos(owner, x, y, z, r);
      if (owner < 0 || owner > count || seen[owner]++)
        throw std::runtime_error("GHOST_BACKEND_INSERTION: invalid augmented identity");
      ++stored;
    } while (loop.inc());
  };
  if constexpr (std::is_base_of_v<voro::container_periodic_base, ContainerT>) {
    voro::c_loop_all_periodic all(con);
    verify(all);
  } else {
    voro::c_loop_all all(con);
    verify(all);
  }
  if (stored != count + 1)
    throw std::runtime_error("GHOST_BACKEND_INSERTION: missing augmented storage");

  voro::voronoicell_neighbor cell;
  auto compute = [&](auto& selected) -> py::dict {
    if (!selected.start() || selected.inc())
      throw std::runtime_error("GHOST_BACKEND_INSERTION: selected slot missing");
    int owner;
    double x, y, z, r;
    selected.pos(owner, x, y, z, r);
    if (owner != count || !seen[count])
      throw std::runtime_error("GHOST_BACKEND_INSERTION: selected identity differs");
    py::dict result;
    if (con.compute_cell(cell, selected)) {
      result = build_cell_dict(cell, -1, x, y, z, opts);
      // The selected private identity may occur on periodic self faces. It
      // never becomes a public adjacent_cell, even on geometry-only routes.
      if (opts.faces)
        for (py::handle item : result["faces"].cast<py::list>()) {
          py::dict face = py::reinterpret_borrow<py::dict>(item);
          if (face["adjacent_cell"].cast<int>() == count)
            face.attr("pop")("adjacent_cell");
        }
      result["empty"] = false;
      result["query_index"] = query_index;
    } else {
      result = build_empty_ghost_dict(query_index, x, y, z, opts);
    }
    return result;
  };
  if constexpr (std::is_base_of_v<voro::container_periodic_base, ContainerT>) {
    voro::c_loop_order_periodic selected(con, selected_order);
    return compute(selected);
  } else {
    voro::c_loop_order selected(con, selected_order);
    return compute(selected);
  }
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  pyvoro2::native_runtime::bind_inspection(m);
  m.doc() = "pyvoro2 core bindings (Voro++)";
  pyvoro2::native_witness::register_bindings(m);
  m.attr("_EAGER_ALLOCATION_LIMIT_BYTES") =
      py::int_(native::eager_allocation_limit_bytes);
  pyvoro2::native_runtime::guarded_def(m, "_test_checked_count", [](py::ssize_t value) {
    return native::checked_int(value, "test count");
  }, py::arg());
  pyvoro2::native_runtime::guarded_def(m, "_test_checked_int_add", [](int lhs, int rhs) {
    return native::checked_int_add(lhs, rhs, "test addition");
  }, py::arg(), py::arg());
  pyvoro2::native_runtime::guarded_def(m, "_test_checked_int_multiply", [](int lhs, int rhs) {
    return native::checked_int_multiply(lhs, rhs, "test multiplication");
  }, py::arg(), py::arg());
  pyvoro2::native_runtime::guarded_def(m, "_test_allocation_estimate", [](std::size_t bytes) {
    native::ByteEstimate estimate;
    estimate.add_bytes(bytes, "test byte accumulation");
    estimate.enforce_limit();
    return estimate.total();
  }, py::arg());
  pyvoro2::native_runtime::guarded_def(m,
      "_test_rectangular_safety_candidate_count",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         std::array<std::array<double, 2>, 3> bounds,
         std::array<bool, 3> periodic,
         py::array_t<double, py::array::c_style | py::array::forcecast>
             inserted_queries) {
        native::require_matrix_shape(points, 3, "points");
        native::require_matrix_shape(inserted_queries, 3, "queries");
        native::require_finite_array(points, "points");
        native::require_finite_array(inserted_queries, "queries");
        native::require_primary_containment(points, bounds, "points");
        native::require_primary_containment(inserted_queries, bounds,
                                            "queries");
        return native::require_rectangular_duplicate_safety<3>(
            points, bounds, periodic, &inserted_queries);
      },
      py::arg("points"),
      py::arg("bounds"),
      py::arg("periodic"),
      py::arg("inserted_queries"));
  pyvoro2::native_runtime::guarded_def(m,
      "_test_periodic_safety_candidate_count",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         std::array<double, 6> cell_params,
         py::array_t<double, py::array::c_style | py::array::forcecast>
             inserted_queries) {
        native::require_matrix_shape(points, 3, "points");
        native::require_matrix_shape(inserted_queries, 3, "queries");
        native::require_finite_array(points, "points");
        native::require_finite_array(inserted_queries, "queries");
        native::require_periodic_primary_containment(points, cell_params,
                                                     "points");
        native::require_periodic_primary_containment(
            inserted_queries, cell_params, "queries");
        return native::require_periodic_duplicate_safety(
            points, cell_params, &inserted_queries);
      },
      py::arg("points"),
      py::arg("cell_params"),
      py::arg("inserted_queries"));
  pyvoro2::native_runtime::guarded_def(m,
      "_test_periodic_safety_certificate",
      [](std::array<double, 6> cell_params) {
        const native::PeriodicSafetyGeometry geometry =
            native::periodic_safety_geometry(cell_params);
        return py::make_tuple(geometry.coefficient_bound_upper,
                              geometry.bins);
      },
      py::arg("cell_params"));
  pyvoro2::native_runtime::guarded_def(m,
      "_test_periodic_safety_keys",
      [](std::array<double, 3> point,
         std::array<double, 6> cell_params) {
        const native::PeriodicSafetyGeometry geometry =
            native::periodic_safety_geometry(cell_params);
        return native::periodic_safety_keys(point.data(), geometry);
      },
      py::arg("point"),
      py::arg("cell_params"));
  pyvoro2::native_runtime::guarded_def(m,
      "_test_periodic_pair_is_unsafe",
      [](std::array<double, 3> left,
         std::array<double, 3> right,
         std::array<double, 6> cell_params) {
        const native::PeriodicSafetyGeometry geometry =
            native::periodic_safety_geometry(cell_params);
        return native::periodic_pair_is_unsafe(
            left.data(), right.data(), geometry);
      },
      py::arg("left"),
      py::arg("right"),
      py::arg("cell_params"));
  pyvoro2::native_runtime::guarded_def(m,
      "_test_periodic_key_alias_budget",
      [](std::vector<std::size_t> key_counts) {
        std::size_t cumulative_aliases = 0;
        for (const std::size_t key_count : key_counts) {
          native::count_triclinic_key_aliases(
              key_count, cumulative_aliases);
        }
        return cumulative_aliases;
      },
      py::arg("key_counts"));
  pyvoro2::native_runtime::guarded_def(m,
      "_test_periodic_resource_estimate",
      [](std::array<double, 6> cell_params,
         std::array<int, 3> blocks,
         int init_mem,
         int particle_stride) {
        native::require_positive_controls(blocks.data(), blocks.size(),
                                          init_mem);
        for (std::size_t i = 0; i < cell_params.size(); ++i) {
          if (!std::isfinite(cell_params[i])) {
            native::fail("cell_params[" + std::to_string(i) + "]",
                         "must be finite");
          }
        }
        for (const std::size_t i : {std::size_t{0}, std::size_t{2},
                                    std::size_t{5}}) {
          if (!(cell_params[i] > 0.0)) {
            native::fail("cell_params[" + std::to_string(i) + "]",
                         "must be a positive finite periodic length");
          }
        }
        const native::Periodic3DResourceEstimate estimate =
            native::estimate_periodic_3d_resources(
                cell_params, blocks, init_mem, particle_stride);
        py::dict result;
        result["primary_blocks"] = estimate.primary_blocks;
        result["ey_bound"] = estimate.ey_bound;
        result["ez_bound"] = estimate.ez_bound;
        result["oy"] = estimate.oy;
        result["oz"] = estimate.oz;
        result["extended_blocks"] = estimate.extended_blocks;
        result["hx"] = estimate.hx;
        result["hy"] = estimate.hy;
        result["hz"] = estimate.hz;
        result["mask_size"] = estimate.mask_size;
        result["queue_size"] = estimate.queue_size;
        result["known_eager_bytes"] =
            estimate.known_eager_allocation.total();
        return result;
      },
      py::arg("cell_params"),
      py::arg("blocks"),
      py::arg("init_mem"),
      py::arg("particle_stride"));

  pyvoro2::native_runtime::guarded_def(m,
      "compute_box_standard",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         std::array<std::array<double, 2>, 3> bounds,
         std::array<int, 3> blocks,
         std::array<bool, 3> periodic,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        native::preflight_box<3>(points, ids, nullptr, bounds, blocks,
                                 periodic, init_mem, 3);
        const auto n = points.shape(0);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();

        container con(bounds[0][0],
                      bounds[0][1],
                      bounds[1][0],
                      bounds[1][1],
                      bounds[2][0],
                      bounds[2][1],
                      blocks[0],
                      blocks[1],
                      blocks[2],
                      periodic[0],
                      periodic[1],
                      periodic[2],
                      init_mem);

        for (py::ssize_t i = 0; i < n; i++) {
          con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
        }

        c_loop_all loop(con);
        return compute_cells_impl(con, loop, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 3>{false, false, false},
      py::arg("init_mem"),
      py::arg("opts"));

  pyvoro2::native_runtime::guarded_def(m,
      "compute_box_power",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         py::array_t<double, py::array::c_style | py::array::forcecast> radii,
         std::array<std::array<double, 2>, 3> bounds,
         std::array<int, 3> blocks,
         std::array<bool, 3> periodic,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        native::preflight_box<3>(points, ids, &radii, bounds, blocks,
                                 periodic, init_mem, 4);
        const auto n = points.shape(0);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();
        auto r = radii.unchecked<1>();

        container_poly con(bounds[0][0],
                           bounds[0][1],
                           bounds[1][0],
                           bounds[1][1],
                           bounds[2][0],
                           bounds[2][1],
                           blocks[0],
                           blocks[1],
                           blocks[2],
                           periodic[0],
                           periodic[1],
                           periodic[2],
                           init_mem);

        for (py::ssize_t i = 0; i < n; i++) {
          con.put(id(i), p(i, 0), p(i, 1), p(i, 2), r(i));
        }

        c_loop_all loop(con);
        return compute_cells_impl(con, loop, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("radii"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 3>{false, false, false},
      py::arg("init_mem"),
      py::arg("opts"));

  // Periodic cell variants
  pyvoro2::native_runtime::guarded_def(m,
      "compute_periodic_standard",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         std::array<double, 6> cell_params,
         std::array<int, 3> blocks,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        native::preflight_periodic_3d(points, ids, nullptr, cell_params,
                                      blocks, init_mem, 3);
        const auto n = points.shape(0);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();

        container_periodic con(cell_params[0],
                               cell_params[1],
                               cell_params[2],
                               cell_params[3],
                               cell_params[4],
                               cell_params[5],
                               blocks[0],
                               blocks[1],
                               blocks[2],
                               init_mem);

        for (py::ssize_t i = 0; i < n; i++) {
          con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
        }

        c_loop_all_periodic loop(con);
        return compute_cells_impl(con, loop, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("cell_params"),
      py::arg("blocks"),
      py::arg("init_mem"),
      py::arg("opts"));

  pyvoro2::native_runtime::guarded_def(m,
      "compute_periodic_power",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         py::array_t<double, py::array::c_style | py::array::forcecast> radii,
         std::array<double, 6> cell_params,
         std::array<int, 3> blocks,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        native::preflight_periodic_3d(points, ids, &radii, cell_params,
                                      blocks, init_mem, 4);
        const auto n = points.shape(0);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();
        auto r = radii.unchecked<1>();

        container_periodic_poly con(cell_params[0],
                                    cell_params[1],
                                    cell_params[2],
                                    cell_params[3],
                                    cell_params[4],
                                    cell_params[5],
                                    blocks[0],
                                    blocks[1],
                                    blocks[2],
                                    init_mem);

        for (py::ssize_t i = 0; i < n; i++) {
          con.put(id(i), p(i, 0), p(i, 1), p(i, 2), r(i));
        }

        c_loop_all_periodic loop(con);
        return compute_cells_impl(con, loop, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("radii"),
      py::arg("cell_params"),
      py::arg("blocks"),
      py::arg("init_mem"),
      py::arg("opts"));

// Batch point-location queries (find_voronoi_cell)
pyvoro2::native_runtime::guarded_def(m,
    "locate_box_standard",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       std::array<std::array<double, 2>, 3> bounds,
       std::array<int, 3> blocks,
       std::array<bool, 3> periodic,
       int init_mem,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries, bool return_source) -> py::tuple {
      pyvoro2::locate_source::profile(return_source);
      native::preflight_box<3>(points, ids, nullptr, bounds, blocks, periodic,
                               init_mem, 3, &queries);
      const auto n = points.shape(0);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);

      container con(bounds[0][0],
                    bounds[0][1],
                    bounds[1][0],
                    bounds[1][1],
                    bounds[2][0],
                    bounds[2][1],
                    blocks[0],
                    blocks[1],
                    blocks[2],
                    periodic[0],
                    periodic[1],
                    periodic[2],
                    init_mem);

      auto source = pyvoro2::locate_source::rectangle<3>(con, points, ids, nullptr, queries, bounds, blocks, periodic, return_source);
      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
      }

      pyvoro2::locate_source::verify_population(con, points, ids, nullptr, source, source.expected_blocks, source.storage_blocks);

      py::array_t<bool> found_arr(m);
      py::array_t<int> pid_arr(m);
      py::array_t<double> pos_arr({m, py::ssize_t(3)});

      auto found = found_arr.mutable_unchecked<1>();
      auto pid_out = pid_arr.mutable_unchecked<1>();
      auto pos_out = pos_arr.mutable_unchecked<2>();

      const double nan = std::numeric_limits<double>::quiet_NaN();

      for (py::ssize_t i = 0; i < m; i++) {
        double rx = nan, ry = nan, rz = nan;
        int pid = -1;
        pyvoro2::locate_source::guard_query(source, queries, static_cast<int>(i), false);
        const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), q(i, 2), rx, ry, rz, pid);
        found(i) = ok;
        pid_out(i) = ok ? pid : -1;
        pos_out(i, 0) = rx;
        pos_out(i, 1) = ry;
        pos_out(i, 2) = rz;
      }

      if (return_source) return py::make_tuple(found_arr, pid_arr, pos_arr, source.packet());
      return py::make_tuple(found_arr, pid_arr, pos_arr);
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("bounds"),
    py::arg("blocks"),
    py::arg("periodic") = std::array<bool, 3>{false, false, false},
    py::arg("init_mem"),
    py::arg("queries"), py::arg("return_source") = false);

pyvoro2::native_runtime::guarded_def(m,
    "locate_box_power",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       py::array_t<double, py::array::c_style | py::array::forcecast> radii,
       std::array<std::array<double, 2>, 3> bounds,
       std::array<int, 3> blocks,
       std::array<bool, 3> periodic,
       int init_mem,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries, bool return_source) -> py::tuple {
      pyvoro2::locate_source::profile(return_source);
      native::preflight_box<3>(points, ids, &radii, bounds, blocks, periodic,
                               init_mem, 4, &queries);
      const auto n = points.shape(0);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto r = radii.unchecked<1>();
      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);

      container_poly con(bounds[0][0],
                         bounds[0][1],
                         bounds[1][0],
                         bounds[1][1],
                         bounds[2][0],
                         bounds[2][1],
                         blocks[0],
                         blocks[1],
                         blocks[2],
                         periodic[0],
                         periodic[1],
                         periodic[2],
                         init_mem);

      auto source = pyvoro2::locate_source::rectangle<3>(con, points, ids, &radii, queries, bounds, blocks, periodic, return_source);
      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2), r(i));
      }

      pyvoro2::locate_source::verify_population(con, points, ids, &radii, source, source.expected_blocks, source.storage_blocks);

      py::array_t<bool> found_arr(m);
      py::array_t<int> pid_arr(m);
      py::array_t<double> pos_arr({m, py::ssize_t(3)});

      auto found = found_arr.mutable_unchecked<1>();
      auto pid_out = pid_arr.mutable_unchecked<1>();
      auto pos_out = pos_arr.mutable_unchecked<2>();

      const double nan = std::numeric_limits<double>::quiet_NaN();

      for (py::ssize_t i = 0; i < m; i++) {
        double rx = nan, ry = nan, rz = nan;
        int pid = -1;
        pyvoro2::locate_source::guard_query(source, queries, static_cast<int>(i), false);
        const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), q(i, 2), rx, ry, rz, pid);
        found(i) = ok;
        pid_out(i) = ok ? pid : -1;
        pos_out(i, 0) = rx;
        pos_out(i, 1) = ry;
        pos_out(i, 2) = rz;
      }

      if (return_source) return py::make_tuple(found_arr, pid_arr, pos_arr, source.packet());
      return py::make_tuple(found_arr, pid_arr, pos_arr);
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("radii"),
    py::arg("bounds"),
    py::arg("blocks"),
    py::arg("periodic") = std::array<bool, 3>{false, false, false},
    py::arg("init_mem"),
    py::arg("queries"), py::arg("return_source") = false);

pyvoro2::native_runtime::guarded_def(m,
    "locate_periodic_standard",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       std::array<double, 6> cell_params,
       std::array<int, 3> blocks,
       int init_mem,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries, bool return_source) -> py::tuple {
      pyvoro2::locate_source::profile(return_source);
      native::preflight_periodic_3d(points, ids, nullptr, cell_params, blocks,
                                    init_mem, 3, &queries);
      const auto n = points.shape(0);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);

      container_periodic con(cell_params[0],
                             cell_params[1],
                             cell_params[2],
                             cell_params[3],
                             cell_params[4],
                             cell_params[5],
                             blocks[0],
                             blocks[1],
                             blocks[2],
                             init_mem);

      auto source = pyvoro2::locate_source::triclinic(con, points, ids, nullptr, queries, return_source);
      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
      }

      pyvoro2::locate_source::verify_population(con, points, ids, nullptr, source, source.expected_blocks, source.storage_blocks);

      py::array_t<bool> found_arr(m);
      py::array_t<int> pid_arr(m);
      py::array_t<double> pos_arr({m, py::ssize_t(3)});

      auto found = found_arr.mutable_unchecked<1>();
      auto pid_out = pid_arr.mutable_unchecked<1>();
      auto pos_out = pos_arr.mutable_unchecked<2>();

      const double nan = std::numeric_limits<double>::quiet_NaN();

      for (py::ssize_t i = 0; i < m; i++) {
        double rx = nan, ry = nan, rz = nan;
        int pid = -1;
        pyvoro2::locate_source::guard_query(source, queries, static_cast<int>(i), true);
        const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), q(i, 2), rx, ry, rz, pid);
        found(i) = ok;
        pid_out(i) = ok ? pid : -1;
        pos_out(i, 0) = rx;
        pos_out(i, 1) = ry;
        pos_out(i, 2) = rz;
      }

      if (return_source) return py::make_tuple(found_arr, pid_arr, pos_arr, source.packet());
      return py::make_tuple(found_arr, pid_arr, pos_arr);
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("cell_params"),
    py::arg("blocks"),
    py::arg("init_mem"),
    py::arg("queries"), py::arg("return_source") = false);

pyvoro2::native_runtime::guarded_def(m,
    "locate_periodic_power",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       py::array_t<double, py::array::c_style | py::array::forcecast> radii,
       std::array<double, 6> cell_params,
       std::array<int, 3> blocks,
       int init_mem,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries, bool return_source) -> py::tuple {
      pyvoro2::locate_source::profile(return_source);
      native::preflight_periodic_3d(points, ids, &radii, cell_params, blocks,
                                    init_mem, 4, &queries);
      const auto n = points.shape(0);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto r = radii.unchecked<1>();
      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);

      container_periodic_poly con(cell_params[0],
                                  cell_params[1],
                                  cell_params[2],
                                  cell_params[3],
                                  cell_params[4],
                                  cell_params[5],
                                  blocks[0],
                                  blocks[1],
                                  blocks[2],
                                  init_mem);

      auto source = pyvoro2::locate_source::triclinic(con, points, ids, &radii, queries, return_source);
      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2), r(i));
      }

      pyvoro2::locate_source::verify_population(con, points, ids, &radii, source, source.expected_blocks, source.storage_blocks);

      py::array_t<bool> found_arr(m);
      py::array_t<int> pid_arr(m);
      py::array_t<double> pos_arr({m, py::ssize_t(3)});

      auto found = found_arr.mutable_unchecked<1>();
      auto pid_out = pid_arr.mutable_unchecked<1>();
      auto pos_out = pos_arr.mutable_unchecked<2>();

      const double nan = std::numeric_limits<double>::quiet_NaN();

      for (py::ssize_t i = 0; i < m; i++) {
        double rx = nan, ry = nan, rz = nan;
        int pid = -1;
        pyvoro2::locate_source::guard_query(source, queries, static_cast<int>(i), true);
        const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), q(i, 2), rx, ry, rz, pid);
        found(i) = ok;
        pid_out(i) = ok ? pid : -1;
        pos_out(i, 0) = rx;
        pos_out(i, 1) = ry;
        pos_out(i, 2) = rz;
      }

      if (return_source) return py::make_tuple(found_arr, pid_arr, pos_arr, source.packet());
      return py::make_tuple(found_arr, pid_arr, pos_arr);
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("radii"),
    py::arg("cell_params"),
    py::arg("blocks"),
    py::arg("init_mem"),
    py::arg("queries"), py::arg("return_source") = false);


// Batch ghost-cell computations (compute_ghost_cell)
pyvoro2::native_runtime::guarded_def(m,
    "ghost_box_standard",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       std::array<std::array<double, 2>, 3> bounds,
       std::array<int, 3> blocks,
       std::array<bool, 3> periodic,
       int init_mem,
       std::tuple<bool, bool, bool> opts_tuple,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
      native::preflight_box<3>(points, ids, nullptr, bounds, blocks, periodic,
                               init_mem, 3, &queries, nullptr, true, false);
      const auto n = points.shape(0);

      const auto opts = parse_opts(opts_tuple);

      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);
      native::checked_int_add(static_cast<int>(n), 1, "augmented ghost count");

      py::list out;
      for (py::ssize_t i = 0; i < m; i++) {
        try {
          container con(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1],
                        bounds[2][0], bounds[2][1], blocks[0], blocks[1],
                        blocks[2], periodic[0], periodic[1], periodic[2], init_mem);
          out.append(safe_selected_ghost(con, points, ids, nullptr,
                                        {q(i, 0), q(i, 1), q(i, 2)}, 0.0,
                                        static_cast<int>(i), opts));
        } catch (const std::exception& error) {
          ghost_bridge_failure(static_cast<int>(i), error);
        }
      }

      return out;
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("bounds"),
    py::arg("blocks"),
    py::arg("periodic") = std::array<bool, 3>{false, false, false},
    py::arg("init_mem"),
    py::arg("opts"),
    py::arg("queries"));

pyvoro2::native_runtime::guarded_def(m,
    "ghost_box_power",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       py::array_t<double, py::array::c_style | py::array::forcecast> radii,
       std::array<std::array<double, 2>, 3> bounds,
       std::array<int, 3> blocks,
       std::array<bool, 3> periodic,
       int init_mem,
       std::tuple<bool, bool, bool> opts_tuple,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries,
       py::array_t<double, py::array::c_style | py::array::forcecast> ghost_radii) {
      native::preflight_box<3>(points, ids, &radii, bounds, blocks, periodic,
                               init_mem, 4, &queries, &ghost_radii, true,
                               false);
      const auto n = points.shape(0);
      const py::ssize_t m = queries.shape(0);

      const auto opts = parse_opts(opts_tuple);

      auto q = queries.unchecked<2>();
      auto gr = ghost_radii.unchecked<1>();
      native::checked_int_add(static_cast<int>(n), 1, "augmented ghost count");

      py::list out;
      for (py::ssize_t i = 0; i < m; i++) {
        try {
          container_poly con(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1],
                             bounds[2][0], bounds[2][1], blocks[0], blocks[1],
                             blocks[2], periodic[0], periodic[1], periodic[2], init_mem);
          out.append(safe_selected_ghost(con, points, ids, &radii,
                                        {q(i, 0), q(i, 1), q(i, 2)}, gr(i),
                                        static_cast<int>(i), opts));
        } catch (const std::exception& error) {
          ghost_bridge_failure(static_cast<int>(i), error);
        }
      }

      return out;
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("radii"),
    py::arg("bounds"),
    py::arg("blocks"),
    py::arg("periodic") = std::array<bool, 3>{false, false, false},
    py::arg("init_mem"),
    py::arg("opts"),
    py::arg("queries"),
    py::arg("ghost_radii"));

// Periodic container variants
pyvoro2::native_runtime::guarded_def(m,
    "ghost_periodic_standard",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       std::array<double, 6> cell_params,
       std::array<int, 3> blocks,
       int init_mem,
       std::tuple<bool, bool, bool> opts_tuple,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
      native::preflight_periodic_3d(points, ids, nullptr, cell_params, blocks,
                                    init_mem, 3, &queries, nullptr, true,
                                    false);
      const auto n = points.shape(0);

      const auto opts = parse_opts(opts_tuple);

      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);
      native::checked_int_add(static_cast<int>(n), 1, "augmented ghost count");

      py::list out;

      for (py::ssize_t i = 0; i < m; i++) {
        try {
          container_periodic con(cell_params[0],
                               cell_params[1],
                               cell_params[2],
                               cell_params[3],
                               cell_params[4],
                               cell_params[5],
                               blocks[0],
                               blocks[1],
                               blocks[2],
                               init_mem);

          out.append(safe_selected_ghost(con, points, ids, nullptr,
                                        {q(i, 0), q(i, 1), q(i, 2)}, 0.0,
                                        static_cast<int>(i), opts));
        } catch (const std::exception& error) {
          ghost_bridge_failure(static_cast<int>(i), error);
        }
      }

      return out;
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("cell_params"),
    py::arg("blocks"),
    py::arg("init_mem"),
    py::arg("opts"),
    py::arg("queries"));

pyvoro2::native_runtime::guarded_def(m,
    "ghost_periodic_power",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       py::array_t<double, py::array::c_style | py::array::forcecast> radii,
       std::array<double, 6> cell_params,
       std::array<int, 3> blocks,
       int init_mem,
       std::tuple<bool, bool, bool> opts_tuple,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries,
       py::array_t<double, py::array::c_style | py::array::forcecast> ghost_radii) {
      native::preflight_periodic_3d(points, ids, &radii, cell_params, blocks,
                                    init_mem, 4, &queries, &ghost_radii, true,
                                    false);
      const auto n = points.shape(0);
      const py::ssize_t m = queries.shape(0);

      const auto opts = parse_opts(opts_tuple);

      auto q = queries.unchecked<2>();
      auto gr = ghost_radii.unchecked<1>();
      native::checked_int_add(static_cast<int>(n), 1, "augmented ghost count");

      py::list out;

      for (py::ssize_t i = 0; i < m; i++) {
        try {
          container_periodic_poly con(cell_params[0],
                                    cell_params[1],
                                    cell_params[2],
                                    cell_params[3],
                                    cell_params[4],
                                    cell_params[5],
                                    blocks[0],
                                    blocks[1],
                                    blocks[2],
                                    init_mem);

          out.append(safe_selected_ghost(con, points, ids, &radii,
                                        {q(i, 0), q(i, 1), q(i, 2)}, gr(i),
                                        static_cast<int>(i), opts));
        } catch (const std::exception& error) {
          ghost_bridge_failure(static_cast<int>(i), error);
        }
      }

      return out;
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("radii"),
    py::arg("cell_params"),
    py::arg("blocks"),
    py::arg("init_mem"),
    py::arg("opts"),
    py::arg("queries"),
    py::arg("ghost_radii"));

}
