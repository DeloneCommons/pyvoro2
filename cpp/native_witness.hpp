#ifndef PYVORO2_NATIVE_WITNESS_HPP
#define PYVORO2_NATIVE_WITNESS_HPP

#include "voro++.hh"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pybind11 { class module_; }

namespace pyvoro2::native_witness {

// Private transport IDs are occurrence IDs, never particle IDs. The backend
// copies them with its existing per-directed-edge neighbor bookkeeping.
struct Origin {
  int token;
  const char* kind;
  int owner;
  std::array<double, 3> normal;
  double offset;
  int legacy_owner;
  int axis = -1;
  int sense = 0;
  bool periodic = false;
  int side = 0;
};

struct WitnessFace {
  std::vector<int> vertices;
  int token;
  std::vector<int> edge_tokens;
  int legacy_owner;
};

inline bool same_bits(double lhs, double rhs) {
  return std::memcmp(&lhs, &rhs, sizeof(double)) == 0;
}

inline int checked_next_token(std::size_t count) {
  if (count >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("native witness provenance token overflow");
  return static_cast<int>(count) + 1;
}

inline void require_same_geometry(voro::voronoicell_base& actual,
                                  voro::voronoicell_base& replay) {
  if (actual.p != replay.p)
    throw std::runtime_error("native witness noninterference: vertex count");
  for (int i = 0; i < actual.p; ++i) {
    if (actual.nu[i] != replay.nu[i])
      throw std::runtime_error("native witness noninterference: vertex order");
    for (int j = 0; j < 3; ++j)
      if (!same_bits(actual.pts[4 * i + j], replay.pts[4 * i + j]))
        throw std::runtime_error("native witness noninterference: vertex bits");
    for (int j = 0; j < 2 * actual.nu[i] + 1; ++j)
      if (actual.ed[i][j] != replay.ed[i][j])
        throw std::runtime_error("native witness noninterference: edge topology");
  }
}

class ObservedCell : public voro::voronoicell_neighbor {
 public:
  std::vector<Origin> origins;

  ObservedCell() = default;
  explicit ObservedCell(double maximum_squared_length)
      : voro::voronoicell_neighbor(maximum_squared_length), replay_(true) {}
  ObservedCell(const ObservedCell&) = delete;
  ObservedCell(ObservedCell&&) = delete;

  void begin_box(int owner, const std::array<bool, 3>& periodic) {
    owner_ = owner;
    periodic_ = periodic;
    seed_ = nullptr;
    replay_ = false;
  }

  void begin_periodic(int owner, ObservedCell& seed) {
    owner_ = owner;
    seed_ = &seed;
    replay_ = false;
  }

  ObservedCell& operator=(ObservedCell& other) {
    if (this != &other) {
      voro::voronoicell_neighbor::operator=(other);
      origins = other.origins;
      owner_ = other.owner_;
      periodic_ = other.periodic_;
      // Completed copies own their tags; they never borrow another seed.
      seed_ = nullptr;
      replay_ = other.replay_;
    }
    return *this;
  }

  // container_periodic_base assigns its real, non-neighbor unit_voro here.
  // Copy that actual geometry, then import provenance only after exact identity
  // with the source-coupled replay has been established. In particular, do not
  // replace the native seed's geometry or receiver's native tolerances.
  void operator=(voro::voronoicell& actual_seed) {
    if (seed_ == nullptr)
      throw std::runtime_error("native witness missing triclinic seed replay");
    require_same_geometry(actual_seed, *seed_);
    voro::voronoicell_neighbor::operator=(actual_seed);
    origins = seed_->origins;
    for (Origin& origin : origins)
      if (std::strcmp(origin.kind, "triclinic_seed") == 0)
        origin.owner = owner_;
    for (int i = 0; i < p; ++i)
      for (int j = 0; j < nu[i]; ++j) {
        const int token = seed_->ne[i][j];
        if (std::strcmp(origin(token).kind, "triclinic_seed") != 0)
          throw std::runtime_error("native witness unbounded triclinic seed");
        ne[i][j] = token;
      }
    seed_ = nullptr;
  }

  void init(double xmin, double xmax, double ymin, double ymax,
            double zmin, double zmax) {
    voro::voronoicell_neighbor::init(xmin, xmax, ymin, ymax, zmin, zmax);
    origins.clear();
    const std::array<double, 6> limits{xmin, xmax, ymin, ymax, zmin, zmax};
    for (int side = 0; side < 6; ++side) {
      const int axis = side / 2;
      const int sense = side % 2 == 0 ? -1 : 1;
      std::array<double, 3> normal{};
      normal[axis] = sense;
      // Match init_base's actual doubled bound before orienting the row.
      double doubled = limits[side];
      doubled *= 2;
      const double offset = sense < 0 ? -doubled : doubled;
      const bool periodic = !replay_ && periodic_[axis];
      const int legacy = -side - 1;
      append(replay_ ? "construction_bound" : "orthogonal_seed",
             periodic ? owner_ : legacy, normal, offset, legacy,
             axis, sense, periodic, legacy);
    }
    for (int i = 0; i < p; ++i)
      for (int j = 0; j < nu[i]; ++j) ne[i][j] = -ne[i][j];
  }

  bool nplane(double x, double y, double z, double rs, int neighbor) {
    const int token = append("particle", neighbor, {x, y, z}, rs, neighbor);
    return voro::voronoicell_neighbor::nplane(x, y, z, rs, token);
  }

  bool plane(double x, double y, double z) {
    // This expression is the vendored plane(x,y,z) overload's exact source
    // order. The source-coupled unitcell replay calls this overload directly.
    double rsq = x*x+y*y+z*z;
    return plane(x, y, z, rsq);
  }

  bool plane(double x, double y, double z, double rs) {
    const int token = append("triclinic_seed", owner_, {x, y, z}, rs, 0);
    return voro::voronoicell_neighbor::nplane(x, y, z, rs, token);
  }

  const Origin& origin(int token) const {
    if (token <= 0 || static_cast<std::size_t>(token) > origins.size())
      throw std::runtime_error("native witness unknown provenance token");
    return origins[static_cast<std::size_t>(token - 1)];
  }

  std::vector<WitnessFace> witness_faces() const {
    std::vector<std::vector<unsigned char>> visited;
    std::size_t edges = 0;
    for (int i = 0; i < p; ++i) {
      if (nu[i] < 0)
        throw std::runtime_error("native witness invalid vertex order");
      visited.emplace_back(static_cast<std::size_t>(nu[i]), 0);
      edges += static_cast<std::size_t>(nu[i]);
    }
    std::vector<WitnessFace> faces;
    // Use native face_vertices' start order while avoiding its temporary
    // mutation of ed. Read every directed-edge tag, including the closing one.
    for (int start = 1; start < p; ++start)
      for (int slot = 0; slot < nu[start]; ++slot) {
        if (visited[start][slot]) continue;
        WitnessFace face{{}, ne[start][slot], {}, 0};
        face.legacy_owner = origin(face.token).legacy_owner;
        int vertex = start, edge = slot;
        do {
          if (vertex < 0 || vertex >= p || edge < 0 || edge >= nu[vertex] ||
              visited[vertex][edge] || face.vertices.size() >= edges)
            throw std::runtime_error("native witness invalid face cycle");
          visited[vertex][edge] = 1;
          const int token = ne[vertex][edge];
          const Origin& provenance = origin(token);
          if (std::strcmp(provenance.kind, "construction_bound") == 0)
            throw std::runtime_error("native witness surviving construction bound");
          if (token != face.token)
            throw std::runtime_error("native witness mixed face provenance tokens");
          face.vertices.push_back(vertex);
          face.edge_tokens.push_back(token);
          const int next = ed[vertex][edge];
          if (next < 0 || next >= p)
            throw std::runtime_error("native witness invalid face adjacency");
          const int relation = ed[vertex][nu[vertex] + edge];
          if (relation < 0 || relation >= nu[next] ||
              ed[next][relation] != vertex)
            throw std::runtime_error("native witness invalid edge relation");
          edge = relation == nu[next] - 1 ? 0 : relation + 1;
          vertex = next;
        } while (vertex != start);
        if (edge != slot || face.vertices.size() < 3)
          throw std::runtime_error("native witness invalid face closure");
        faces.push_back(std::move(face));
      }
    for (const auto& vertex : visited)
      for (unsigned char edge : vertex)
        if (!edge)
          throw std::runtime_error("native witness unvisited directed edge");
    return faces;
  }

 private:
  int owner_ = 0;
  std::array<bool, 3> periodic_{};
  ObservedCell* seed_ = nullptr;
  bool replay_ = false;

  int append(const char* kind, int owner, const std::array<double, 3>& normal,
             double offset, int legacy_owner, int axis = -1, int sense = 0,
             bool periodic = false, int side = 0) {
    if (!std::isfinite(offset) || !std::isfinite(normal[0]) ||
        !std::isfinite(normal[1]) || !std::isfinite(normal[2]))
      throw std::runtime_error("native witness non-finite native plane");
    const int token = checked_next_token(origins.size());
    origins.push_back(Origin{token, kind, owner, normal, offset, legacy_owner,
                             axis, sense, periodic, side});
    return token;
  }
};

void register_bindings(pybind11::module_& module);

}  // namespace pyvoro2::native_witness

#endif
