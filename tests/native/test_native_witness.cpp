#include "native_witness.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pyvoro2::native_witness::ObservedCell;

// These checks also run in the Release builds used by wheel CI.
#define CHECK(expression)                                                   \
  do {                                                                      \
    if (!(expression))                                                      \
      throw std::runtime_error(std::string(__func__) + ":" +                 \
                               std::to_string(__LINE__) + ": " #expression); \
  } while (false)

template <class Function>
void expect_failure(Function function, const char* diagnostic) {
  try {
    function();
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()).find(diagnostic) == std::string::npos)
      throw std::runtime_error(std::string("expected ") + diagnostic +
                               "; got " + error.what());
    return;
  }
  throw std::runtime_error(std::string("missing failure: ") + diagnostic);
}

bool same_bits(double first, double second) {
  return std::memcmp(&first, &second, sizeof(double)) == 0;
}

int edge_to(const ObservedCell& cell, int vertex, int neighbor) {
  for (int edge = 0; edge < cell.nu[vertex]; ++edge)
    if (cell.ed[vertex][edge] == neighbor) return edge;
  throw std::runtime_error("missing face edge");
}

void check_native(ObservedCell& observed, voro::voronoicell_neighbor& native) {
  CHECK(observed.p == native.p);
  for (int vertex = 0; vertex < native.p; ++vertex) {
    CHECK(observed.nu[vertex] == native.nu[vertex]);
    for (int axis = 0; axis < 3; ++axis)
      CHECK(same_bits(observed.pts[4 * vertex + axis],
                      native.pts[4 * vertex + axis]));
    for (int edge = 0; edge < 2 * native.nu[vertex] + 1; ++edge)
      CHECK(observed.ed[vertex][edge] == native.ed[vertex][edge]);
    for (int edge = 0; edge < native.nu[vertex]; ++edge)
      CHECK(observed.origin(observed.ne[vertex][edge]).legacy_owner ==
            native.ne[vertex][edge]);
  }
  CHECK(same_bits(observed.volume(), native.volume()));

  std::vector<int> native_faces, native_owners;
  native.face_vertices(native_faces);
  native.neighbors(native_owners);
  const auto faces = observed.witness_faces();
  CHECK(faces.size() == native_owners.size());
  std::size_t position = 0, covered_edges = 0;
  for (std::size_t index = 0; index < faces.size(); ++index) {
    const auto& face = faces[index];
    CHECK(native_faces[position++] == static_cast<int>(face.vertices.size()));
    CHECK(face.legacy_owner == native_owners[index]);
    CHECK(face.edge_tokens.size() == face.vertices.size());
    for (std::size_t edge = 0; edge < face.vertices.size(); ++edge) {
      CHECK(native_faces[position++] == face.vertices[edge]);
      CHECK(face.edge_tokens[edge] == face.token);
      const int vertex = face.vertices[edge];
      const int next = face.vertices[(edge + 1) % face.vertices.size()];
      CHECK(observed.ne[vertex][edge_to(observed, vertex, next)] == face.token);
      ++covered_edges;
    }
  }
  CHECK(position == native_faces.size());
  std::size_t native_edges = 0;
  for (int vertex = 0; vertex < native.p; ++vertex)
    native_edges += native.nu[vertex];
  CHECK(covered_edges == native_edges);
}

void initialize(ObservedCell& observed, voro::voronoicell_neighbor& native,
                double extent = 1) {
  observed.begin_box(4, {true, false, false});
  observed.init(-extent, extent, -extent, extent, -extent, extent);
  native.init(-extent, extent, -extent, extent, -extent, extent);
  check_native(observed, native);
}

bool cut(ObservedCell& observed, voro::voronoicell_neighbor& native,
         double x, double y, double z, double offset, int owner) {
  const bool expected = native.nplane(x, y, z, offset, owner);
  const bool actual = observed.nplane(x, y, z, offset, owner);
  CHECK(actual == expected);
  if (actual) check_native(observed, native);
  return actual;
}

bool survives(const ObservedCell& cell, int token) {
  for (const auto& face : cell.witness_faces())
    if (face.token == token) return true;
  return false;
}

void occurrence_tokens() {
  ObservedCell cell;
  voro::voronoicell_neighbor native;
  initialize(cell, native);
  CHECK(cell.origins.size() == 6);
  CHECK(cell.origins[0].owner == 4);
  CHECK(cell.origins[2].owner == -3);
  const double offset = std::nextafter(1.0, 2.0);
  CHECK(cut(cell, native, 1, 0, 0, offset, 7));
  const int first = cell.origins.back().token;
  CHECK(same_bits(cell.origins.back().offset, offset));
  CHECK(survives(cell, first));
  CHECK(cut(cell, native, 1, 0, 0, offset, 7));
  const int repeated = cell.origins.back().token;
  CHECK(repeated != first);
  CHECK(survives(cell, first) != survives(cell, repeated));
  CHECK(cut(cell, native, 1, 0, 0, offset, 19));
  const int coincident = cell.origins.back().token;
  CHECK(coincident != repeated && coincident != first);
  CHECK(cell.origins.back().owner == 19);
  // The native marginal-cut path may replace the prior face tag. It must
  // select one recorded occurrence, while preserving the ordinary owner.
  CHECK(static_cast<int>(survives(cell, first)) +
        static_cast<int>(survives(cell, repeated)) +
        static_cast<int>(survives(cell, coincident)) == 1);
  const int surviving = survives(cell, first) ? first :
                        (survives(cell, repeated) ? repeated : coincident);
  CHECK(cut(cell, native, 0, 1, 0, 3, 7));
  CHECK(!survives(cell, cell.origins.back().token));
  CHECK(cut(cell, native, -1, 0, 0, 0.5, cell.origin(surviving).owner));
  CHECK(cell.origins.back().token != surviving);
  CHECK(survives(cell, surviving));
  CHECK(survives(cell, cell.origins.back().token));
  CHECK(cell.origin(surviving).owner == cell.origins.back().owner);
  CHECK(cell.origins.size() == 11);
}

void deep_assignment() {
  ObservedCell assigned;
  voro::voronoicell_neighbor expected;
  {
    ObservedCell source;
    voro::voronoicell_neighbor native;
    initialize(source, native);
    CHECK(cut(source, native, 1, 0, 0, 1, 17));
    assigned = source;
    expected = native;
    CHECK(assigned.origins.data() != source.origins.data());
    CHECK(assigned.ed != source.ed);
    CHECK(assigned.ne != source.ne);
    CHECK(assigned.pts != source.pts);
    CHECK(cut(source, native, 0, 1, 0, 0.5, 23));
    CHECK(assigned.origins.size() + 1 == source.origins.size());
    check_native(assigned, expected);
    assigned = assigned;
    check_native(assigned, expected);
  }
  // The original cell and its neighbor storage have now been destroyed.
  check_native(assigned, expected);
  CHECK(cut(assigned, expected, 0, 0, 1, 0.25, 29));
  ObservedCell reassigned;
  reassigned = assigned;
  reassigned = assigned;
  check_native(reassigned, expected);
  CHECK(reassigned.origins.data() != assigned.origins.data());
}

void full_cycle_validation() {
  ObservedCell cell;
  voro::voronoicell_neighbor native;
  initialize(cell, native);
  const auto faces = cell.witness_faces();
  const auto& face = faces.front();
  const int vertex = face.vertices.back();
  const int edge = edge_to(cell, vertex, face.vertices.front());
  const int original = cell.ne[vertex][edge];

  // Change only the closing edge: first-edge-only extraction misses this.
  cell.ne[vertex][edge] = faces[1].token;
  expect_failure([&] { cell.witness_faces(); }, "mixed face provenance");
  for (const int invalid : {0, -1, static_cast<int>(cell.origins.size()) + 1}) {
    cell.ne[vertex][edge] = invalid;
    expect_failure([&] { cell.witness_faces(); }, "unknown provenance token");
  }
  cell.ne[vertex][edge] = original;
  check_native(cell, native);

  const int adjacency = cell.ed[vertex][edge];
  cell.ed[vertex][edge] = cell.p;
  expect_failure([&] { cell.witness_faces(); }, "invalid face adjacency");
  cell.ed[vertex][edge] = adjacency;
  const int relation = cell.ed[vertex][cell.nu[vertex] + edge];
  cell.ed[vertex][cell.nu[vertex] + edge] = cell.nu[adjacency];
  expect_failure([&] { cell.witness_faces(); }, "invalid edge relation");
  cell.ed[vertex][cell.nu[vertex] + edge] = relation;
  check_native(cell, native);
}

void construction_bounds() {
  ObservedCell seed(100);
  voro::voronoicell actual(100);
  seed.init(-2, 2, -2, 2, -2, 2);
  actual.init(-2, 2, -2, 2, -2, 2);
  CHECK(std::string(seed.origins.front().kind) == "construction_bound");
  expect_failure([&] { seed.witness_faces(); }, "surviving construction bound");
  ObservedCell receiver;
  expect_failure([&] { receiver = actual; }, "missing triclinic seed replay");
  receiver.begin_periodic(42, seed);
  expect_failure([&] { receiver = actual; }, "unbounded triclinic seed");
  for (int axis = 0; axis < 3; ++axis)
    for (int sign : {-1, 1}) {
      double normal[3] = {0, 0, 0};
      normal[axis] = sign;
      CHECK(seed.plane(normal[0], normal[1], normal[2], 2));
      CHECK(actual.plane(normal[0], normal[1], normal[2], 2));
    }
  CHECK(seed.witness_faces().size() == 6);
  receiver = actual;
  CHECK(same_bits(receiver.volume(), actual.volume()));
  CHECK(receiver.origins.data() != seed.origins.data());
  for (const auto& face : receiver.witness_faces()) {
    CHECK(std::string(receiver.origin(face.token).kind) == "triclinic_seed");
    CHECK(receiver.origin(face.token).owner == 42);
    CHECK(seed.origin(face.token).owner == 0);
    CHECK(face.legacy_owner == 0);
  }
  // An unmatched native seed must fail before importing its provenance.
  CHECK(actual.plane(1, 0, 0, 0.5));
  receiver.begin_periodic(42, seed);
  expect_failure([&] { receiver = actual; }, "noninterference");
}

void token_exhaustion() {
  using pyvoro2::native_witness::checked_next_token;
  const auto maximum = static_cast<std::size_t>(std::numeric_limits<int>::max());
  CHECK(checked_next_token(0) == 1);
  CHECK(checked_next_token(maximum - 1) == std::numeric_limits<int>::max());
  for (const auto size : {maximum, maximum + 1,
                          std::numeric_limits<std::size_t>::max()})
    expect_failure([&] { checked_next_token(size); }, "token overflow");
}

void higher_order_and_marginal() {
  ObservedCell cell;
  voro::voronoicell_neighbor native;
  initialize(cell, native, 2);
  int owner = 0;
  for (int x : {-1, 1})
    for (int y : {-1, 1})
      for (int z : {-1, 1})
        CHECK(cut(cell, native, x, y, z, 2, owner++));
  // The eight exact halfspaces form an octahedron with six order-four vertices.
  CHECK(cell.p == 6);
  CHECK(cell.witness_faces().size() == 8);
  for (int vertex = 0; vertex < cell.p; ++vertex) CHECK(cell.nu[vertex] == 4);
  CHECK(cut(cell, native, 1, 1, 0, 0, owner++));
  CHECK(cut(cell, native, 1, 0, 0, 0, owner++));
  CHECK(cut(cell, native, 0, 1, 0, 0, owner++));
  // Each cut contains existing vertices or edges exactly. Their removals and
  // marginal-vertex updates must leave complete, single-token face cycles.
  for (int vertex = 0; vertex < cell.p; ++vertex) CHECK(cell.nu[vertex] >= 3);
  CHECK(!cut(cell, native, 0, 0, 1, -2, owner));
  CHECK(cell.origins.back().owner == owner);
  CHECK(cell.origins.back().offset == -2);
}

void memory_relocation() {
  ObservedCell cell;
  voro::voronoicell_neighbor native;
  initialize(cell, native, 2);
  const int original_capacity = cell.current_vertices;
  int relocations = 0;
  constexpr int count = 320;
  const double pi = std::acos(-1.0);
  const double angle_step = pi * (3 - std::sqrt(5.0));
  for (int index = 0; index < count; ++index) {
    // Deterministic Fibonacci sphere tangents, all supporting the unit ball.
    const double z = 1 - 2 * (index + 0.5) / count;
    const double radius = std::sqrt(1 - z * z);
    const double angle = index * angle_step;
    const int capacity = cell.current_vertices;
    CHECK(cut(cell, native, radius * std::cos(angle),
              radius * std::sin(angle), z, 2, index % 11));
    if (cell.current_vertices > capacity) ++relocations;
  }
  CHECK(cell.p > 256);
  CHECK(cell.current_vertices > original_capacity);
  CHECK(relocations > 0);
  CHECK(cell.witness_faces().size() == count);
  CHECK(cell.origins.size() == count + 6);
  // Copying a grown pool must also own all relocated neighbor allocations.
  ObservedCell copy;
  copy = cell;
  check_native(copy, native);
  CHECK(copy.ne != cell.ne);
  CHECK(cut(copy, native, 1, 0, 0, 0, 99));
  CHECK(cell.origins.size() + 1 == copy.origins.size());
  CHECK(cell.witness_faces().size() == count);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string name = argc == 2 ? argv[1] : "";
  try {
    if (name == "occurrence_tokens") occurrence_tokens();
    else if (name == "deep_assignment") deep_assignment();
    else if (name == "full_cycle_validation") full_cycle_validation();
    else if (name == "construction_bounds") construction_bounds();
    else if (name == "token_exhaustion") token_exhaustion();
    else if (name == "higher_order_and_marginal") higher_order_and_marginal();
    else if (name == "memory_relocation") memory_relocation();
    else throw std::runtime_error("unknown native witness test: " + name);
    std::cout << "native witness: " << name << " passed\n";
  } catch (const std::exception& error) {
    std::cerr << "native witness: " << name << " failed: " << error.what() << '\n';
    return 1;
  }
}
