find_package(MPI REQUIRED)
enable_language(Fortran)

set(_tests test_callbacks_p2p test_callbacks_multi test_future test_sendrecv)

foreach (_t ${_tests})
  add_executable(${_t} ${_t}.f90)
  target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/bindings/ ${MPI_Fortran_INCLUDE_DIRS})
  target_link_libraries(${_t} ghex ${MPI_Fortran_LIBRARIES})
endforeach ()
