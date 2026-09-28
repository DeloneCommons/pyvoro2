// Eager, arithmetic-free raw FP inspection before lazy numerical imports.
// Loader metadata work can itself round: public routes retain this module's
// checker from package initialization and consult it before importing _core.
#include "native_runtime.hpp"

PYBIND11_MODULE(_fpguard, module) {
  pyvoro2::native_runtime::bind_inspection(module);
}
