# The ordinary 2D producer and observer share the existing target-local strict
# FP policy.  No 3D geometric witness/replay machinery is used.
function(pyvoro2_planar_witness target_name)
  pyvoro2_native_fp(${target_name})
  if(MSVC)
    target_link_libraries(${target_name} PRIVATE pybind11::windows_extras)
  endif()

  # This conservative closure includes all linked 2D routes as well as the
  # ordinary adapter and its effective build controls.  The approved digest is
  # a separately reviewed literal, not the just-computed digest itself.
  file(GLOB _sources CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "vendor/voro++/2d/src/*")
  list(APPEND _sources cpp/bindings2d.cpp cpp/native_preconditions.hpp
    cpp/native_fp_contract.hpp cpp/planar_witness.cpp cpp/planar_witness.hpp
    CMakeLists.txt cmake/NativeFP.cmake cmake/PlanarWitness.cmake pyproject.toml)
  list(SORT _sources)
  set(_stream "")
  foreach(_path IN LISTS _sources)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/${_path}")
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${_path}" _digest)
    string(APPEND _stream "${_path}:${_digest}\n")
  endforeach()
  string(SHA256 _source_sha "${_stream}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/planar-source-closure.txt" "${_stream}")

  # Source-local flags are emitted after target options.  Refuse that unchecked
  # override route rather than inferring contraction/LTO from a version string.
  get_target_property(_target_sources ${target_name} SOURCES)
  foreach(_source IN LISTS _target_sources)
    get_source_file_property(_source_options "${_source}" COMPILE_OPTIONS)
    get_source_file_property(_source_flags "${_source}" COMPILE_FLAGS)
    if(_source_options OR _source_flags)
      message(FATAL_ERROR "WP6 source-local compile flags require separate qualification: ${_source}")
    endif()
  endforeach()
  option(PYVORO2_PLANAR_QUALIFICATION "Build private WP6 stock comparison probes" OFF)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _config)
  get_target_property(_options ${target_name} COMPILE_OPTIONS)
  get_target_property(_link_options ${target_name} LINK_OPTIONS)
  set(_build "${CMAKE_SYSTEM_NAME}|${CMAKE_SYSTEM_PROCESSOR}|${CMAKE_CXX_COMPILER_ID}|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_CXX_FLAGS}|${CMAKE_BUILD_TYPE}|${CMAKE_CXX_FLAGS_${_config}}|${_options}|${_link_options}|IPO=OFF|CXX17|qualification=${PYVORO2_PLANAR_QUALIFICATION}")
  string(SHA256 _build_sha "${_build}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/planar-build-profile.txt" "${_build}\n")
  target_compile_definitions(${target_name} PRIVATE
    PYVORO2_PLANAR_SOURCE_SHA256="${_source_sha}"
    PYVORO2_PLANAR_BUILD_SHA256="${_build_sha}"
    PYVORO2_PLANAR_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    PYVORO2_PLANAR_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    PYVORO2_PLANAR_FP_CONTROLLED=1)

  # Standalone qualification builds expose stock internal coordinates, so B/C
  # checks do not rely on rounded public coordinates.  Ordinary wheels omit it.
  if(PYVORO2_PLANAR_QUALIFICATION)
    target_compile_definitions(${target_name} PRIVATE PYVORO2_PLANAR_QUALIFICATION=1)
  endif()
endfunction()
