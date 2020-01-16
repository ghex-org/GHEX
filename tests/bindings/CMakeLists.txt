find_package(OpenMP REQUIRED)
enable_language(Fortran)

# set(_tests test_callbacks_p2p test_callbacks_multi test_future test_sendrecv)
set(_tests test_context test_communicator test_message test_send_recv_ft test_send_recv_cb)

foreach (_t ${_tests})
  add_executable(${_t} ${_t}.f90)
  target_compile_options(${_t} PRIVATE ${OpenMP_CXX_FLAGS})
  target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/bindings/ ${MPI_Fortran_INCLUDE_DIRS})
  target_link_libraries(${_t} PRIVATE fhex ${MPI_Fortran_LIBRARIES} OpenMP::OpenMP_CXX)
endforeach ()
