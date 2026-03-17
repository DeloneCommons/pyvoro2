#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "voro++_2d.hh"

namespace py = pybind11;
using namespace voro;

namespace {

struct OutputOpts {
  bool vertices;
  bool adjacency;
  bool edges;
};

OutputOpts parse_opts(const std::tuple<bool, bool, bool>& opts) {
  return OutputOpts{std::get<0>(opts), std::get<1>(opts), std::get<2>(opts)};
}

void check_points(const py::array_t<double>& points) {
  if (points.ndim() != 2 || points.shape(1) != 2) {
    throw py::value_error("points must have shape (n, 2)");
  }
}

void check_ids(const py::array_t<int>& ids, py::ssize_t n) {
  if (ids.ndim() != 1 || ids.shape(0) != n) {
    throw py::value_error("ids must have shape (n,)");
  }
}

void check_radii(const py::array_t<double>& radii, py::ssize_t n) {
  if (radii.ndim() != 1 || radii.shape(0) != n) {
    throw py::value_error("radii must have shape (n,)");
  }
}

void check_queries(const py::array_t<double>& queries) {
  if (queries.ndim() != 2 || queries.shape(1) != 2) {
    throw py::value_error("queries must have shape (m, 2)");
  }
}

void check_ghost_radii(
    const py::array_t<double>& ghost_radii,
    py::ssize_t m
) {
  if (ghost_radii.ndim() != 1 || ghost_radii.shape(0) != m) {
    throw py::value_error("ghost_radii must have shape (m,)");
  }
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

py::dict build_empty_ghost_dict(
    int query_index,
    double x,
    double y,
    const OutputOpts& opts
) {
  py::dict out;
  out["id"] = -1;
  out["empty"] = true;
  out["area"] = 0.0;

  py::list site;
  site.append(x);
  site.append(y);
  out["site"] = site;
  out["query_index"] = query_index;

  if (opts.vertices) out["vertices"] = py::list();
  if (opts.adjacency) out["adjacency"] = py::list();
  if (opts.edges) out["edges"] = py::list();
  return out;
}

template <class ContainerT>
py::list compute_cells_impl(ContainerT& con, const OutputOpts& opts) {
  py::list out;
  voronoicell_neighbor_2d cell;
  c_loop_all_2d loop(con);

  if (loop.start()) {
    do {
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

template <class ContainerT>
bool append_ghost_cell(
    ContainerT& con,
    int ghost_id,
    int query_index,
    double x,
    double y,
    const OutputOpts& opts,
    py::list& out
) {
  c_loop_all_2d loop(con);
  voronoicell_neighbor_2d cell;
  if (loop.start()) {
    do {
      if (loop.pid() != ghost_id) {
        continue;
      }
      if (con.compute_cell(cell, loop)) {
        py::dict d = build_cell_dict(cell, -1, x, y, opts);
        d["empty"] = false;
        d["query_index"] = query_index;
        out.append(d);
      } else {
        out.append(build_empty_ghost_dict(query_index, x, y, opts));
      }
      return true;
    } while (loop.inc());
  }
  return false;
}

}  // namespace

PYBIND11_MODULE(_core2d, m) {
  m.doc() = "pyvoro2 planar core bindings (legacy 2D Voro++)";

  m.def(
      "compute_box_standard",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         std::array<std::array<double, 2>, 2> bounds,
         std::array<int, 2> blocks,
         std::array<bool, 2> periodic,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        check_points(points);
        const auto n = points.shape(0);
        check_ids(ids, n);
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

        return compute_cells_impl(con, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 2>{false, false},
      py::arg("init_mem"),
      py::arg("opts"));

  m.def(
      "compute_box_power",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         py::array_t<double, py::array::c_style | py::array::forcecast> radii,
         std::array<std::array<double, 2>, 2> bounds,
         std::array<int, 2> blocks,
         std::array<bool, 2> periodic,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        check_points(points);
        const auto n = points.shape(0);
        check_ids(ids, n);
        check_radii(radii, n);
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

  m.def(
      "locate_box_standard",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         std::array<std::array<double, 2>, 2> bounds,
         std::array<int, 2> blocks,
         std::array<bool, 2> periodic,
         int init_mem,
         py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
        check_points(points);
        const auto n = points.shape(0);
        check_ids(ids, n);
        check_queries(queries);

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

        for (py::ssize_t i = 0; i < n; ++i) {
          con.put(id(i), p(i, 0), p(i, 1));
        }

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
          const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), rx, ry, pid);
          found(i) = ok;
          pid_out(i) = ok ? pid : -1;
          pos_out(i, 0) = rx;
          pos_out(i, 1) = ry;
        }

        return py::make_tuple(found_arr, pid_arr, pos_arr);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 2>{false, false},
      py::arg("init_mem"),
      py::arg("queries"));

  m.def(
      "locate_box_power",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         py::array_t<double, py::array::c_style | py::array::forcecast> radii,
         std::array<std::array<double, 2>, 2> bounds,
         std::array<int, 2> blocks,
         std::array<bool, 2> periodic,
         int init_mem,
         py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
        check_points(points);
        const auto n = points.shape(0);
        check_ids(ids, n);
        check_radii(radii, n);
        check_queries(queries);

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

        for (py::ssize_t i = 0; i < n; ++i) {
          con.put(id(i), p(i, 0), p(i, 1), r(i));
        }

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
          const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), rx, ry, pid);
          found(i) = ok;
          pid_out(i) = ok ? pid : -1;
          pos_out(i, 0) = rx;
          pos_out(i, 1) = ry;
        }

        return py::make_tuple(found_arr, pid_arr, pos_arr);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("radii"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 2>{false, false},
      py::arg("init_mem"),
      py::arg("queries"));

  m.def(
      "ghost_box_standard",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         std::array<std::array<double, 2>, 2> bounds,
         std::array<int, 2> blocks,
         std::array<bool, 2> periodic,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple,
         py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
        check_points(points);
        const auto n = points.shape(0);
        check_ids(ids, n);
        check_queries(queries);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();
        auto q = queries.unchecked<2>();
        const py::ssize_t m_q = queries.shape(0);
        const int ghost_id = std::numeric_limits<int>::max();

        py::list out;
        for (py::ssize_t qi = 0; qi < m_q; ++qi) {
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

          const double x = q(qi, 0);
          const double y = q(qi, 1);
          con.put(ghost_id, x, y);
          if (!append_ghost_cell(con, ghost_id, static_cast<int>(qi), x, y, opts, out)) {
            out.append(build_empty_ghost_dict(static_cast<int>(qi), x, y, opts));
          }
        }

        return out;
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 2>{false, false},
      py::arg("init_mem"),
      py::arg("opts"),
      py::arg("queries"));

  m.def(
      "ghost_box_power",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         py::array_t<double, py::array::c_style | py::array::forcecast> radii,
         std::array<std::array<double, 2>, 2> bounds,
         std::array<int, 2> blocks,
         std::array<bool, 2> periodic,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple,
         py::array_t<double, py::array::c_style | py::array::forcecast> queries,
         py::array_t<double, py::array::c_style | py::array::forcecast> ghost_radii) {
        check_points(points);
        const auto n = points.shape(0);
        check_ids(ids, n);
        check_radii(radii, n);
        check_queries(queries);
        const py::ssize_t m_q = queries.shape(0);
        check_ghost_radii(ghost_radii, m_q);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();
        auto r = radii.unchecked<1>();
        auto q = queries.unchecked<2>();
        auto gr = ghost_radii.unchecked<1>();
        const int ghost_id = std::numeric_limits<int>::max();

        py::list out;
        for (py::ssize_t qi = 0; qi < m_q; ++qi) {
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

          const double x = q(qi, 0);
          const double y = q(qi, 1);
          con.put(ghost_id, x, y, gr(qi));
          if (!append_ghost_cell(con, ghost_id, static_cast<int>(qi), x, y, opts, out)) {
            out.append(build_empty_ghost_dict(static_cast<int>(qi), x, y, opts));
          }
        }

        return out;
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("radii"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 2>{false, false},
      py::arg("init_mem"),
      py::arg("opts"),
      py::arg("queries"),
      py::arg("ghost_radii"));
}
