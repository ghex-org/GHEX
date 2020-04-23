find_package(OpenMP REQUIRED)
enable_language(Fortran)

set(_tests test_f_context test_f_communicator test_f_message test_f_send_recv_ft test_f_send_recv_cb test_f_structured test_f_cubed_sphere test_f_cubed_sphere_os)

foreach (_t ${_tests})
  add_executable(${_t} ${_t}.f90)
  target_compile_options(${_t} PRIVATE ${OpenMP_CXX_FLAGS})
  target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/bindings/ ${MPI_Fortran_INCLUDE_DIRS})
  target_link_libraries(${_t} PRIVATE fhex ${MPI_Fortran_LIBRARIES} OpenMP::OpenMP_CXX)
endforeach ()
