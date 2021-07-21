add_executable(sizes sizes.cpp)
target_link_libraries(sizes PRIVATE ghexlib)

add_library(fhex SHARED communicator_bind.cpp context_bind.cpp message_bind.cpp future_bind.cpp request_bind.cpp structured_staged_bind.cpp cubed_sphere_bind.cpp ghex_comm_mod.f90 ghex_message_mod.f90 ghex_future_mod.f90 ghex_request_mod.f90 ghex_structured_mod.f90 ghex_cubed_sphere_mod.f90 ghex_defs.f90 ghex_mod.f90 ghex_config.h)

if (GHEX_FORTRAN_OPENMP)
   find_package(OpenMP REQUIRED)
   target_link_libraries(fhex PRIVATE OpenMP::OpenMP_Fortran)
endif ()

if (GHEX_USE_UCP)
   target_compile_definitions(fhex PRIVATE GHEX_USE_UCP)
   find_package(Threads REQUIRED)
   target_link_libraries(fhex INTERFACE Threads::Threads)
endif()

if (GHEX_USE_XPMEM)
  target_compile_definitions(fhex PRIVATE GHEX_USE_XPMEM)
endif()

if(${GHEX_FORTRAN_FP} STREQUAL "float")
  set(GHEX_FORTRAN_FP_KIND 4)
else()
  set(GHEX_FORTRAN_FP_KIND 8)
endif()

message(STATUS "GHEX_FORTRAN_FP_KIND ${GHEX_FORTRAN_FP_KIND}")
target_compile_definitions(fhex PRIVATE GHEX_FORTRAN_FP=${GHEX_FORTRAN_FP} GHEX_FORTRAN_FP_KIND=${GHEX_FORTRAN_FP_KIND})
target_include_directories(fhex PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(fhex PRIVATE ghexlib)
target_link_libraries(fhex INTERFACE MPI::MPI_Fortran)

add_custom_command(OUTPUT ghex_config.h
  COMMAND "${CMAKE_CURRENT_BINARY_DIR}/sizes" ">ghex_config.h"
  DEPENDS sizes)

install(TARGETS fhex
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_comm_mod.mod
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_cubed_sphere_mod.mod
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_defs.mod
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_future_mod.mod
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_message_mod.mod
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_mod.mod
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_request_mod.mod
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_structured_mod.mod
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/)
