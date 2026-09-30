cmake_minimum_required(VERSION 3.20)
project(pyvoro2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
include(cmake/NativeQualification.cmake)

# Use the real linker directly in controlled Windows builds. Ninja supplies
# parallelism; cl.exe /MP would hide independent compile attribution. The
# recorder retains LINK's manifest, then directly observes SDK mt.exe embedding
# its exact metadata. /MANIFEST:EMBED hides rc/cvtres input invocations.
if(MSVC AND DEFINED ENV{PYVORO2_EVIDENCE_DIR})
  set(CMAKE_CXX_CREATE_SHARED_MODULE
    "<CMAKE_LINKER> /nologo <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> /DLL /MANIFEST /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\" /OUT:<TARGET> /IMPLIB:<TARGET_IMPLIB> /PDB:<TARGET_PDB> <OBJECTS> <LINK_LIBRARIES>")
endif()

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

# NO_EXTRAS excludes pybind11's implicit LTO; NativeFP also disables CMake IPO.
pybind11_add_module(_core NO_EXTRAS
  cpp/bindings.cpp
  cpp/native_witness.cpp
  ${VORO_SOURCES}
)
if(MSVC)
  target_compile_options(_core PRIVATE /bigobj)
endif()

include(cmake/NativeFP.cmake)
pyvoro2_native_fp(_core)
# This small raw inspector is preloaded without loading either geometry
# backend. A forward first call must check controls before CPython's lazy
# extension import (whose filesystem timestamps can involve FP conversion).
pybind11_add_module(_fpguard NO_EXTRAS cpp/fpguard.cpp)
pyvoro2_native_fp(_fpguard)
set(_PYVORO2_WITNESS_SOURCE_HASHES "")
foreach(_source unitcell.hh unitcell.cc cell.hh cell.cc v_compute.hh v_compute.cc
                rad_option.hh container.hh container.cc container_prd.hh
                container_prd.cc v_base.hh v_base.cc c_loops.hh)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${VORO_SRC_DIR}/${_source}")
  file(SHA256 "${VORO_SRC_DIR}/${_source}" _source_hash)
  string(APPEND _PYVORO2_WITNESS_SOURCE_HASHES "${_source}:${_source_hash};")
endforeach()
# Semicolons are CMake list delimiters; use commas within the single metadata
# string passed to the compiler.
string(REPLACE ";" "," _PYVORO2_WITNESS_SOURCE_HASHES
  "${_PYVORO2_WITNESS_SOURCE_HASHES}")
set(_PYVORO2_GHOST_SOURCE_HASHES "${_PYVORO2_WITNESS_SOURCE_HASHES}")
file(GLOB _PYVORO2_GHOST_VENDOR_SOURCES CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "vendor/voro++/src/*")
list(SORT _PYVORO2_GHOST_VENDOR_SOURCES)
foreach(_source IN LISTS _PYVORO2_GHOST_VENDOR_SOURCES)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${_source}" _source_hash)
  string(APPEND _PYVORO2_GHOST_SOURCE_HASHES "${_source}:${_source_hash},")
endforeach()
foreach(_source cpp/bindings.cpp cpp/native_witness.cpp cpp/native_witness.hpp
                cpp/native_preconditions.hpp cpp/native_fp_contract.hpp cpp/locate_source.hpp
                CMakeLists.txt cmake/NativeFP.cmake pyproject.toml)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${_source}" _source_hash)
  string(APPEND _PYVORO2_GHOST_SOURCE_HASHES "${_source}:${_source_hash},")
endforeach()
string(SHA256 _PYVORO2_GHOST_SOURCE_SHA256
  "${_PYVORO2_GHOST_SOURCE_HASHES}")
set_source_files_properties(cpp/native_witness.cpp PROPERTIES
  COMPILE_DEFINITIONS
  "PYVORO2_WITNESS_SOURCE_SHA256=\"${_PYVORO2_WITNESS_SOURCE_HASHES}\";PYVORO2_WITNESS_COMPILER_VERSION=\"${CMAKE_CXX_COMPILER_VERSION}\";PYVORO2_WITNESS_COMPILER_ID=\"${CMAKE_CXX_COMPILER_ID}\";PYVORO2_GHOST_SOURCE_SHA256=\"${_PYVORO2_GHOST_SOURCE_SHA256}\"")

target_include_directories(_core PRIVATE
  "${VORO_SRC_DIR}"
)

list(REMOVE_ITEM VORO2D_SOURCES "${VORO2D_SRC_DIR}/v_compute_2d.cc")
pybind11_add_module(_core2d NO_EXTRAS
  cpp/planar_witness.cpp
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
configure_pyvoro_module(_fpguard)
pyvoro2_native_qualification(_core)
pyvoro2_native_qualification(_core2d)
pyvoro2_native_qualification(_fpguard)
include(cmake/PlanarWitness.cmake)
pyvoro2_planar_witness(_core2d)

install(TARGETS _core _core2d _fpguard DESTINATION pyvoro2)
