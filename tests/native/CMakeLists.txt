cmake_minimum_required(VERSION 3.20)
project(pyvoro2_native_witness_tests LANGUAGES CXX)

set(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
set(VORO_SRC "${REPO_ROOT}/vendor/voro++/src")
add_executable(test_native_witness
  test_native_witness.cpp
  "${VORO_SRC}/cell.cc"
  "${VORO_SRC}/common.cc"
)
target_include_directories(test_native_witness PRIVATE
  "${REPO_ROOT}/cpp" "${VORO_SRC}"
)
target_compile_features(test_native_witness PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(test_native_witness PRIVATE /EHsc)
endif()
include("${REPO_ROOT}/cmake/NativeFP.cmake")
pyvoro2_native_fp(test_native_witness)

# WP7 standalone stock-kernel reference and explicit poisoned-slot probe.
add_executable(test_wp7_ghost_reference
  test_wp7_ghost_reference.cpp
  "${VORO_SRC}/c_loops.cc"
  "${VORO_SRC}/cell.cc"
  "${VORO_SRC}/common.cc"
  "${VORO_SRC}/container.cc"
  "${VORO_SRC}/container_prd.cc"
  "${VORO_SRC}/unitcell.cc"
  "${VORO_SRC}/v_base.cc"
  "${VORO_SRC}/v_compute.cc"
  "${VORO_SRC}/wall.cc"
)
target_include_directories(test_wp7_ghost_reference PRIVATE "${VORO_SRC}")
target_compile_features(test_wp7_ghost_reference PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(test_wp7_ghost_reference PRIVATE /EHsc)
endif()
pyvoro2_native_fp(test_wp7_ghost_reference)

option(PYVORO2_NATIVE_TEST_SANITIZERS "Check native witness memory safety" OFF)
if(PYVORO2_NATIVE_TEST_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "Native ASan/UBSan tests require GCC or Clang")
  endif()
  target_compile_options(test_native_witness PRIVATE
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
  )
  target_link_options(test_native_witness PRIVATE -fsanitize=address,undefined)
  target_compile_options(test_wp7_ghost_reference PRIVATE
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
  )
  target_link_options(test_wp7_ghost_reference PRIVATE -fsanitize=address,undefined)
endif()
