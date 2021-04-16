if(EARLY_BUILD)
  add_executable(sizes sizes.cpp)
  target_link_libraries(sizes PRIVATE ghexlib)
  return()
endif()

message(STATUS "CONFIGURE sizes tool")
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}"
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
  -DEARLY_BUILD=ON
  ${CMAKE_SOURCE_DIR}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  OUTPUT_QUIET)

message(STATUS "BUILD sizes tool")
execute_process(COMMAND ${CMAKE_COMMAND} --build . --target sizes
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  OUTPUT_QUIET)

execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/sizes" GHEX_REQUEST_SIZE OUTPUT_VARIABLE GHEX_REQUEST_SIZE)
execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/sizes" GHEX_FUTURE_SIZE OUTPUT_VARIABLE GHEX_FUTURE_SIZE)
execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/sizes" GHEX_FUTURE_MULTI_SIZE OUTPUT_VARIABLE GHEX_FUTURE_MULTI_SIZE)

message(STATUS "GHEX_REQUEST_SIZE=${GHEX_REQUEST_SIZE}")
message(STATUS "GHEX_FUTURE_SIZE=${GHEX_FUTURE_SIZE}")
message(STATUS "GHEX_FUTURE_MULTI_SIZE=${GHEX_FUTURE_MULTI_SIZE}")

add_library(fhex SHARED communicator_bind.cpp context_bind.cpp message_bind.cpp future_bind.cpp request_bind.cpp structured_staged_bind.cpp cubed_sphere_bind.cpp ghex_comm_mod.f90 ghex_message_mod.f90 ghex_future_mod.f90 ghex_request_mod.f90 ghex_structured_mod.f90 ghex_cubed_sphere_mod.f90 ghex_defs.f90 ghex_mod.f90)

if (GHEX_USE_UCP)
   target_compile_definitions(fhex PRIVATE GHEX_USE_UCP)
   find_package(Threads REQUIRED)
   target_link_libraries(fhex INTERFACE Threads::Threads)
endif()

if (GHEX_USE_XPMEM)
  target_compile_definitions(fhex PRIVATE GHEX_USE_XPMEM)
endif()

target_compile_definitions(fhex PRIVATE
  GHEX_FORTRAN_FP=${GHEX_FORTRAN_FP}
  GHEX_REQUEST_SIZE=${GHEX_REQUEST_SIZE}
  GHEX_FUTURE_SIZE=${GHEX_FUTURE_SIZE}
  GHEX_FUTURE_MULTI_SIZE=${GHEX_FUTURE_MULTI_SIZE})
target_include_directories(fhex PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(fhex PRIVATE ghexlib)
target_link_libraries(fhex INTERFACE MPI::MPI_Fortran)
