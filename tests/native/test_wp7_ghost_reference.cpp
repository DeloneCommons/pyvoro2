#include "voro++.hh"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// A defined initialized stock selected-cell reference and an explicit
// poisoned-unused-ID-slot probe of the legacy ghost template. This records
// which legacy paths consume the slot, without relying on undefined integers.
namespace {
constexpr int poison = 0x5a37b1;
constexpr int ghost = 17;

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool has_owner(voro::voronoicell_neighbor& cell, int owner) {
  std::vector<int> owners;
  cell.neighbors(owners);
  for (int value : owners) if (value == owner) return true;
  return false;
}

void same_geometry(voro::voronoicell_neighbor& left,
                   voro::voronoicell_neighbor& right) {
  check(left.p == right.p, "selected/poisoned vertex count");
  for (int i = 0; i < left.p; ++i) {
    check(left.nu[i] == right.nu[i], "selected/poisoned vertex order");
    for (int axis = 0; axis < 3; ++axis)
      check(std::memcmp(&left.pts[4*i+axis], &right.pts[4*i+axis],
                        sizeof(double)) == 0, "selected/poisoned vertex bits");
    for (int j = 0; j < 2*left.nu[i]+1; ++j)
      check(left.ed[i][j] == right.ed[i][j], "selected/poisoned topology");
  }
  double volume_left = left.volume(), volume_right = right.volume();
  check(std::memcmp(&volume_left, &volume_right, sizeof(double)) == 0,
        "selected/poisoned volume bits");
}

template <bool Power, class Container>
void box(bool periodic) {
  Container old(0,1,0,1,0,1,1,1,1,periodic,false,false,1);
  Container defined(0,1,0,1,0,1,1,1,1,periodic,false,false,1);
  old.id[0][old.co[0]] = poison;
  voro::voronoicell_neighbor old_cell, defined_cell;
  bool old_ok, defined_ok;
  if constexpr (Power) {
    old_ok = old.compute_ghost_cell(old_cell,.5,.5,.5,.2);
  } else {
    old_ok = old.compute_ghost_cell(old_cell,.5,.5,.5);
  }
  voro::particle_order ordering(2);
  if constexpr (Power) defined.put(ordering,ghost,.5,.5,.5,.2);
  else defined.put(ordering,ghost,.5,.5,.5);
  check(defined.co[0] == 1 && defined.id[0][0] == ghost,
        "initialized box storage");
  voro::c_loop_order selected(defined, ordering);
  check(selected.start(), "selected box slot");
  defined_ok = defined.compute_cell(defined_cell, selected);
  check(!selected.inc(), "one selected box slot");
  check(old_ok == defined_ok, "box disposition");
  if (old_ok) same_geometry(old_cell, defined_cell);
  check(has_owner(old_cell,poison) == periodic,
        "legacy box poisoned slot read / nonperiodic negative control");
  check(!has_owner(defined_cell,poison), "defined box poison leak");
  if (periodic) check(has_owner(defined_cell,ghost), "defined box self image");
  std::cout << "box " << (Power ? "power" : "standard") << " periodic="
            << periodic << " poison_read=" << has_owner(old_cell,poison)
            << " selected_equal=1\n";
}

template <bool Power, class Container>
void triclinic() {
  Container old(1,.1,1,.2,.1,1,1,1,1,1);
  Container defined(1,.1,1,.2,.1,1,1,1,1,1);
  int primary = old.nx * (old.ey + old.oy * old.ez);
  old.id[primary][old.co[primary]] = poison;
  voro::voronoicell_neighbor old_cell, defined_cell;
  bool old_ok, defined_ok;
  if constexpr (Power) old_ok = old.compute_ghost_cell(old_cell,.5,.5,.5,.2);
  else old_ok = old.compute_ghost_cell(old_cell,.5,.5,.5);
  voro::particle_order ordering(2);
  if constexpr (Power) defined.put(ordering,ghost,.5,.5,.5,.2);
  else defined.put(ordering,ghost,.5,.5,.5);
  voro::c_loop_order_periodic selected(defined, ordering);
  check(selected.start() && selected.ijk == primary,
        "selected triclinic block");
  check(defined.co[primary] == 1 && defined.id[primary][selected.q] == ghost,
        "initialized triclinic storage");
  defined_ok = defined.compute_cell(defined_cell, selected);
  check(!selected.inc(), "one selected triclinic slot");
  check(old_ok == defined_ok, "triclinic disposition");
  if (old_ok) same_geometry(old_cell, defined_cell);
  check(has_owner(old_cell,poison), "legacy triclinic poisoned slot read");
  bool copied_poison = false;
  for (int block = 0; block < old.oxyz; ++block)
    if (block != primary)
      for (int slot = 0; slot < old.co[block]; ++slot)
        copied_poison |= old.id[block][slot] == poison;
  check(copied_poison, "legacy triclinic lazy image copied poisoned ID");
  check(has_owner(defined_cell,ghost), "defined triclinic self image");
  std::cout << "triclinic " << (Power ? "power" : "standard")
            << " poison_read=1 lazy_copy=1 selected_equal=1\n";
}

void selected_growth() {
  voro::container con(0,1,0,1,0,1,1,1,1,true,true,true,1);
  con.put(0,.1,.2,.3);
  con.put(1,.8,.7,.6);
  check(con.mem[0] == 2 && con.co[0] == 2, "persistent growth");
  voro::particle_order ordering(2);
  con.put(ordering,2,.5,.5,.5);
  check(con.mem[0] == 4 && con.co[0] == 3 && con.id[0][2] == 2,
        "initialized ghost after capacity growth");
  voro::c_loop_order selected(con, ordering);
  check(selected.start(), "growth selected slot");
  voro::voronoicell_neighbor cell;
  check(con.compute_cell(cell,selected), "growth selected cell");
  check(has_owner(cell,2), "growth copied self image identity");
  std::cout << "growth initialized_before_read=1 self_image_copy=1\n";
}
}  // namespace

int main() {
  try {
    box<false,voro::container>(false);
    box<true,voro::container_poly>(false);
    box<false,voro::container>(true);
    box<true,voro::container_poly>(true);
    triclinic<false,voro::container_periodic>();
    triclinic<true,voro::container_periodic_poly>();
    selected_growth();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
