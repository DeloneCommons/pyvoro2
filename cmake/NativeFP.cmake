# One arithmetic contract for the 3D producer, observer, and native tests.
# Keep this target-local: the independent 2D backend is outside this repair.
function(pyvoro2_native_fp target_name)
  set_target_properties(${target_name} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION OFF)
  foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_property(TARGET ${target_name} PROPERTY
      INTERPROCEDURAL_OPTIMIZATION_${_config} OFF)
  endforeach()

  set(_guard "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cpp/native_fp_contract.hpp")
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /fp:strict /GL- "/FI${_guard}")
    target_link_options(${target_name} PRIVATE /LTCG:OFF)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_options(${target_name} PRIVATE
      -fno-fast-math -ffp-contract=off -fno-lto "SHELL:-include \"${_guard}\"")
    # Fast-math at link time can install process-wide FTZ/DAZ startup code.
    # Disable raw toolchain LTO flags as well as CMake/pybind11 IPO defaults.
    target_link_options(${target_name} PRIVATE
      -fno-fast-math -ffp-contract=off -fno-lto)
  else()
    message(FATAL_ERROR "The 3D native FP contract requires GCC, Clang, or MSVC")
  endif()
endfunction()
