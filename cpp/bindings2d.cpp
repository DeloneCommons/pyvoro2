#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "voro++_2d.hh"
#include "native_preconditions.hpp"
#include "locate_source.hpp"
#include "planar_witness.hpp"

namespace py = pybind11;
using namespace voro;
namespace native = pyvoro2::native_preconditions;

namespace {

struct OutputOpts {
  bool vertices;
  bool adjacency;
  bool edges;
};

OutputOpts parse_opts(const std::tuple<bool, bool, bool>& opts) {
  return OutputOpts{std::get<0>(opts), std::get<1>(opts), std::get<2>(opts)};
}

py::dict build_cell_dict(
    voronoicell_neighbor_2d& cell,
    int pid,
    double x,
    double y,
    const OutputOpts& opts
) {
  py::dict out;
  out["id"] = pid;
  out["area"] = cell.area();

  py::list site;
  site.append(x);
  site.append(y);
  out["site"] = site;

  if (opts.vertices) {
    std::vector<double> positions;
    cell.vertices(x, y, positions);
    py::list verts;
    for (std::size_t i = 0; i + 1 < positions.size(); i += 2) {
      py::list v;
      v.append(positions[i]);
      v.append(positions[i + 1]);
      verts.append(v);
    }
    out["vertices"] = verts;
  }

  if (opts.adjacency) {
    py::list adj;
    for (int i = 0; i < cell.p; ++i) {
      py::list row;
      row.append(cell.ed[2 * i]);
      row.append(cell.ed[2 * i + 1]);
      adj.append(row);
    }
    out["adjacency"] = adj;
  }

  if (opts.edges) {
    std::vector<int> neigh;
    cell.neighbors(neigh);
    if (neigh.size() != static_cast<std::size_t>(cell.p)) {
      throw std::runtime_error(
          "pyvoro2 internal error: mismatch between planar neighbors and vertices"
      );
    }

    py::list edges;
    for (int i = 0; i < cell.p; ++i) {
      py::dict edge;
      edge["adjacent_cell"] = neigh[static_cast<std::size_t>(i)];
      py::list vids;
      vids.append(i);
      vids.append(cell.ed[2 * i]);
      edge["vertices"] = vids;
      edges.append(edge);
    }
    out["edges"] = edges;
  }

  return out;
}

template <class ContainerT>
py::list compute_cells_impl(ContainerT& con, const OutputOpts& opts) {
  py::list out;
  voronoicell_neighbor_2d cell;
  c_loop_all_2d loop(con);

  if (loop.start()) {
    do {
      pyvoro2::planar_witness::require_selector(
          con, con.p[loop.ij][con.ps * loop.q],
          con.p[loop.ij][con.ps * loop.q + 1], loop.i, loop.j);
      if (con.compute_cell(cell, loop)) {
        int pid;
        double x, y, r;
        loop.pos(pid, x, y, r);
        out.append(build_cell_dict(cell, pid, x, y, opts));
      }
    } while (loop.inc());
  }

  return out;
}

}  // namespace

PYBIND11_MODULE(_core2d, m) {
  pyvoro2::native_runtime::bind_inspection(m);
  pyvoro2::planar_witness::bind(m);
  m.doc() = "pyvoro2 planar core bindings (legacy 2D Voro++)";

  pyvoro2::native_runtime::guarded_def(m,
      "compute_box_standard",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         std::array<std::array<double, 2>, 2> bounds,
         std::array<int, 2> blocks,
         std::array<bool, 2> periodic,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        native::preflight_box<2>(points, ids, nullptr, bounds, blocks,
                                 periodic, init_mem, 2);
        pyvoro2::planar_witness::require_evaluation();
        const auto n = points.shape(0);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();

        container_2d con(bounds[0][0],
                         bounds[0][1],
                         bounds[1][0],
                         bounds[1][1],
                         blocks[0],
                         blocks[1],
                         periodic[0],
                         periodic[1],
                         init_mem);

        for (py::ssize_t i = 0; i < n; ++i) {
          con.put(id(i), p(i, 0), p(i, 1));
        }

        pyvoro2::planar_witness::require_population(con, n);
        return compute_cells_impl(con, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 2>{false, false},
      py::arg("init_mem"),
      py::arg("opts"));

  pyvoro2::native_runtime::guarded_def(m,
      "compute_box_power",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         py::array_t<double, py::array::c_style | py::array::forcecast> radii,
         std::array<std::array<double, 2>, 2> bounds,
         std::array<int, 2> blocks,
         std::array<bool, 2> periodic,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        native::preflight_box<2>(points, ids, &radii, bounds, blocks,
                                 periodic, init_mem, 3);
        pyvoro2::planar_witness::require_evaluation();
        const auto n = points.shape(0);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();
        auto r = radii.unchecked<1>();

        container_poly_2d con(bounds[0][0],
                              bounds[0][1],
                              bounds[1][0],
                              bounds[1][1],
                              blocks[0],
                              blocks[1],
                              periodic[0],
                              periodic[1],
                              init_mem);

        for (py::ssize_t i = 0; i < n; ++i) {
          con.put(id(i), p(i, 0), p(i, 1), r(i));
        }

        pyvoro2::planar_witness::require_population(con, n);
        return compute_cells_impl(con, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("radii"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 2>{false, false},
      py::arg("init_mem"),
      py::arg("opts"));

  pyvoro2::native_runtime::guarded_def(m,
      "locate_box_standard",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         std::array<std::array<double, 2>, 2> bounds,
         std::array<int, 2> blocks,
         std::array<bool, 2> periodic,
         int init_mem,
         py::array_t<double, py::array::c_style | py::array::forcecast> queries, bool return_source) -> py::tuple {
      pyvoro2::locate_source::profile(return_source);
        native::preflight_box<2>(points, ids, nullptr, bounds, blocks,
                                 periodic, init_mem, 2, &queries);
        const auto n = points.shape(0);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();
        auto q = queries.unchecked<2>();
        const py::ssize_t m_q = queries.shape(0);

        container_2d con(bounds[0][0],
                         bounds[0][1],
                         bounds[1][0],
                         bounds[1][1],
                         blocks[0],
                         blocks[1],
                         periodic[0],
                         periodic[1],
                         init_mem);

        auto source = pyvoro2::locate_source::rectangle<2>(con, points, ids, nullptr, queries, bounds, blocks, periodic, return_source);
        for (py::ssize_t i = 0; i < n; ++i) {
          con.put(id(i), p(i, 0), p(i, 1));
        }

        pyvoro2::locate_source::verify_population(con, points, ids, nullptr, source, source.expected_blocks, source.storage_blocks);

        py::array_t<bool> found_arr(m_q);
        py::array_t<int> pid_arr(m_q);
        py::array_t<double> pos_arr({m_q, py::ssize_t(2)});

        auto found = found_arr.mutable_unchecked<1>();
        auto pid_out = pid_arr.mutable_unchecked<1>();
        auto pos_out = pos_arr.mutable_unchecked<2>();

        const double nan = std::numeric_limits<double>::quiet_NaN();

        for (py::ssize_t i = 0; i < m_q; ++i) {
          double rx = nan;
          double ry = nan;
          int pid = -1;
          pyvoro2::locate_source::guard_query(source, queries, static_cast<int>(i), false);
          const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), rx, ry, pid);
          found(i) = ok;
          pid_out(i) = ok ? pid : -1;
          pos_out(i, 0) = rx;
          pos_out(i, 1) = ry;
        }

        if (return_source) return py::make_tuple(found_arr, pid_arr, pos_arr, source.packet());
        return py::make_tuple(found_arr, pid_arr, pos_arr);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 2>{false, false},
      py::arg("init_mem"),
      py::arg("queries"), py::arg("return_source") = false);

  pyvoro2::native_runtime::guarded_def(m,
      "locate_box_power",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         py::array_t<double, py::array::c_style | py::array::forcecast> radii,
         std::array<std::array<double, 2>, 2> bounds,
         std::array<int, 2> blocks,
         std::array<bool, 2> periodic,
         int init_mem,
         py::array_t<double, py::array::c_style | py::array::forcecast> queries, bool return_source) -> py::tuple {
      pyvoro2::locate_source::profile(return_source);
        native::preflight_box<2>(points, ids, &radii, bounds, blocks,
                                 periodic, init_mem, 3, &queries);
        const auto n = points.shape(0);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();
        auto r = radii.unchecked<1>();
        auto q = queries.unchecked<2>();
        const py::ssize_t m_q = queries.shape(0);

        container_poly_2d con(bounds[0][0],
                              bounds[0][1],
                              bounds[1][0],
                              bounds[1][1],
                              blocks[0],
                              blocks[1],
                              periodic[0],
                              periodic[1],
                              init_mem);

        auto source = pyvoro2::locate_source::rectangle<2>(con, points, ids, &radii, queries, bounds, blocks, periodic, return_source);
        for (py::ssize_t i = 0; i < n; ++i) {
          con.put(id(i), p(i, 0), p(i, 1), r(i));
        }

        pyvoro2::locate_source::verify_population(con, points, ids, &radii, source, source.expected_blocks, source.storage_blocks);

        py::array_t<bool> found_arr(m_q);
        py::array_t<int> pid_arr(m_q);
        py::array_t<double> pos_arr({m_q, py::ssize_t(2)});

        auto found = found_arr.mutable_unchecked<1>();
        auto pid_out = pid_arr.mutable_unchecked<1>();
        auto pos_out = pos_arr.mutable_unchecked<2>();

        const double nan = std::numeric_limits<double>::quiet_NaN();

        for (py::ssize_t i = 0; i < m_q; ++i) {
          double rx = nan;
          double ry = nan;
          int pid = -1;
          pyvoro2::locate_source::guard_query(source, queries, static_cast<int>(i), false);
          const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), rx, ry, pid);
          found(i) = ok;
          pid_out(i) = ok ? pid : -1;
          pos_out(i, 0) = rx;
          pos_out(i, 1) = ry;
        }

        if (return_source) return py::make_tuple(found_arr, pid_arr, pos_arr, source.packet());
        return py::make_tuple(found_arr, pid_arr, pos_arr);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("radii"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 2>{false, false},
      py::arg("init_mem"),
      py::arg("queries"), py::arg("return_source") = false);

  // The two ghost entry points are registered by planar_witness::bind above.
  // They share the fresh selected-source route with the witnessed entry points.
#ifndef PYVORO2_PLANAR_QUALIFICATION
  // Only the separately built, non-distributed stock comparison module uses
  // qualification aliases. Every production import freezes its file identity.
  pyvoro2::native_runtime::register_imported_module(m);
#endif
}
