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

option(PYVORO2_NATIVE_TEST_SANITIZERS "Check native witness memory safety" OFF)
if(PYVORO2_NATIVE_TEST_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "Native ASan/UBSan tests require GCC or Clang")
  endif()
  target_compile_options(test_native_witness PRIVATE
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
  )
  target_link_options(test_native_witness PRIVATE -fsanitize=address,undefined)
endif()
