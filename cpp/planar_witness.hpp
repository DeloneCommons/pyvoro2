#ifndef PYVORO2_PLANAR_WITNESS_HPP
#define PYVORO2_PLANAR_WITNESS_HPP

#include <pybind11/pybind11.h>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyvoro2::planar_witness {

[[noreturn]] inline void fail(const char* stage, const char* reason,
                              const std::string& detail) {
  throw std::runtime_error(std::string("planar_certification:") + stage +
                           ":" + reason + ": " + detail);
}

inline bool gradual_underflow() {
  volatile double den = std::numeric_limits<double>::denorm_min();
  volatile double twice = den + den;
  volatile double half = twice * 0.5;
  return twice > den && half == den;
}

inline void require_evaluation() {
  if (std::fegetround() != FE_TONEAREST)
    fail("profile", "rounding", "ordinary planar evaluation requires RNE");
  if (!gradual_underflow())
    fail("profile", "subnormal", "ordinary planar evaluation requires gradual underflow");
}

// Every persistent ID must be stored before clipping can classify a cell as
// hidden.  Stock put() can silently omit a contained point after rounding.
// This also applies when the caller requests no edges or public geometry.
template<class Container>
void require_population(Container& con, pybind11::ssize_t count) {
  std::vector<unsigned char> seen(static_cast<std::size_t>(count), 0);
  pybind11::ssize_t actual = 0;
  for (int block = 0; block < con.nxy; ++block) {
    for (int slot = 0; slot < con.co[block]; ++slot) {
      const int id = con.id[block][slot];
      if (id < 0 || id >= count || seen[id])
        fail("insertion", "population", "stored internal IDs are not one-to-one");
      seen[id] = 1;
      ++actual;
    }
  }
  if (actual != count)
    fail("insertion", "omitted", "a persistent input was not inserted by native put");
}

// The byte-unmodified compute template repeats these literal operations.
// Check them before its unchecked floating-to-integer worklist selection.
template<class Container>
void require_selector(Container& con, double x, double y, int i, int j) {
  double fx, fy;
  con.frac_pos(x, y, i, j, fx, fy);
  for (double value : {fx * con.xsp * 8, fy * con.ysp * 8}) {
    if (!std::isfinite(value) || value <= -1 ||
        value >= std::numeric_limits<int>::max())
      fail("profile", "selector", "unchecked native worklist selector is outside the qualified domain");
    int d = int(value);
    if (d >= 4) { d = 7 - d; if (d < 0) d = 0; }
    if (d < 0 || d >= 4)
      fail("profile", "selector", "native reflected worklist selector is invalid");
  }
}

void bind(pybind11::module_& module);

}  // namespace pyvoro2::planar_witness
#endif
