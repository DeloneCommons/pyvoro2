cmake_minimum_required(VERSION 3.20)
project(pyvoro2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Voro++ sources (vendored)
set(VORO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/voro++/src")
set(VORO2D_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/voro++/2d/src")

set(VORO_SOURCES
  "${VORO_SRC_DIR}/c_loops.cc"
  "${VORO_SRC_DIR}/cell.cc"
  "${VORO_SRC_DIR}/common.cc"
  "${VORO_SRC_DIR}/container.cc"
  "${VORO_SRC_DIR}/container_prd.cc"
  "${VORO_SRC_DIR}/pre_container.cc"
  "${VORO_SRC_DIR}/unitcell.cc"
  "${VORO_SRC_DIR}/v_base.cc"
  "${VORO_SRC_DIR}/v_compute.cc"
  "${VORO_SRC_DIR}/wall.cc"
)

set(VORO2D_SOURCES
  "${VORO2D_SRC_DIR}/common.cc"
  "${VORO2D_SRC_DIR}/cell_2d.cc"
  "${VORO2D_SRC_DIR}/container_2d.cc"
  "${VORO2D_SRC_DIR}/v_base_2d.cc"
  "${VORO2D_SRC_DIR}/v_compute_2d.cc"
  "${VORO2D_SRC_DIR}/c_loops_2d.cc"
  "${VORO2D_SRC_DIR}/wall_2d.cc"
  "${VORO2D_SRC_DIR}/cell_nc_2d.cc"
  "${VORO2D_SRC_DIR}/ctr_boundary_2d.cc"
  "${VORO2D_SRC_DIR}/ctr_quad_2d.cc"
  "${VORO2D_SRC_DIR}/quad_march.cc"
)

pybind11_add_module(_core
  cpp/bindings.cpp
  ${VORO_SOURCES}
)

target_include_directories(_core PRIVATE
  "${VORO_SRC_DIR}"
)

pybind11_add_module(_core2d
  cpp/bindings2d.cpp
  ${VORO2D_SOURCES}
)

target_include_directories(_core2d PRIVATE
  "${VORO2D_SRC_DIR}"
)

function(configure_pyvoro_module target_name)
  if (MSVC)
    target_compile_options(${target_name} PRIVATE /EHsc)
  else()
    target_compile_options(${target_name} PRIVATE -O3)
  endif()

  set(_PYVORO2_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}/pyvoro2")
  set_target_properties(${target_name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_PYVORO2_OUTDIR}"
    RUNTIME_OUTPUT_DIRECTORY "${_PYVORO2_OUTDIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${_PYVORO2_OUTDIR}"
  )

  foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER "${_cfg}" _cfg_uc)
    set_target_properties(${target_name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${_cfg_uc} "${_PYVORO2_OUTDIR}"
      RUNTIME_OUTPUT_DIRECTORY_${_cfg_uc} "${_PYVORO2_OUTDIR}"
      ARCHIVE_OUTPUT_DIRECTORY_${_cfg_uc} "${_PYVORO2_OUTDIR}"
    )
  endforeach()
endfunction()

configure_pyvoro_module(_core)
configure_pyvoro_module(_core2d)

install(TARGETS _core _core2d DESTINATION pyvoro2)
