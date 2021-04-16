add_library(fhex SHARED communicator_bind.cpp context_bind.cpp message_bind.cpp future_bind.cpp request_bind.cpp structured_staged_bind.cpp cubed_sphere_bind.cpp ghex_comm_mod.f90 ghex_message_mod.f90 ghex_future_mod.f90 ghex_request_mod.f90 ghex_structured_mod.f90 ghex_cubed_sphere_mod.f90 ghex_defs.f90 ghex_mod.f90)

if (GHEX_USE_UCP)
   target_compile_definitions(fhex PRIVATE GHEX_USE_UCP)
   find_package(Threads REQUIRED)
   target_link_libraries(fhex INTERFACE Threads::Threads)
endif()

if (GHEX_USE_XPMEM)
  target_compile_definitions(fhex PRIVATE GHEX_USE_XPMEM)
endif()

target_compile_definitions(fhex PRIVATE GHEX_FORTRAN_FP=${GHEX_FORTRAN_FP})
target_include_directories(fhex PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(fhex PRIVATE ghexlib)
target_link_libraries(fhex INTERFACE MPI::MPI_Fortran)
