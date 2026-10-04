# Measurement is consistency data. Only the external controlled finalizer can
# issue a record against the current source manifest and complete build evidence.
if(DEFINED ENV{PYVORO2_EVIDENCE_DIR} AND NOT CMAKE_GENERATOR STREQUAL "Ninja")
  # Visual Studio ignores the compiler/linker launcher properties. The closed
  # observation path requires one compiler process per translation unit.
  message(FATAL_ERROR "Controlled qualification requires the Ninja generator")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/native/qualification/source_policy.py"
    --root "${CMAKE_CURRENT_SOURCE_DIR}"
  RESULT_VARIABLE _qualification_status
  OUTPUT_VARIABLE _qualification_measurement
  ERROR_VARIABLE _qualification_error)
if(NOT _qualification_status EQUAL 0)
  message(FATAL_ERROR "Cannot measure native source closure: ${_qualification_error}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/native-source-measurement.json"
  "${_qualification_measurement}")
foreach(_field source_sha256 schema_sha256 consumer_sha256)
  string(JSON _identity_${_field} GET "${_qualification_measurement}" "${_field}")
endforeach()

function(pyvoro2_native_qualification target_name)
  target_compile_definitions(${target_name} PRIVATE
    PYVORO2_QUALIFICATION_SOURCE_SHA256="${_identity_source_sha256}"
    PYVORO2_QUALIFICATION_SCHEMA_SHA256="${_identity_schema_sha256}"
    PYVORO2_QUALIFICATION_CONSUMER_SHA256="${_identity_consumer_sha256}")
  # This private evidence destination does not grant support or relax flags.
  # Normal builds omit it and have no externally issued qualification record.
  if(DEFINED ENV{PYVORO2_EVIDENCE_DIR})
    if(CMAKE_VERSION VERSION_LESS "3.21")
      message(FATAL_ERROR "Controlled qualification requires CMake >= 3.21")
    endif()
    set(_recorder "${Python_EXECUTABLE};${CMAKE_CURRENT_SOURCE_DIR}/tools/native/qualification/record_command.py;--output-dir;$ENV{PYVORO2_EVIDENCE_DIR};--")
    set_target_properties(${target_name} PROPERTIES
      CXX_COMPILER_LAUNCHER "${_recorder}"
      CXX_LINKER_LAUNCHER "${_recorder}")
  endif()
endfunction()
