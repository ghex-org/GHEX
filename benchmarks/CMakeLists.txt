set(_benchmarks gcl_test_halo_exchange_3D_generic_full comm_2_test_halo_exchange_3D_generic_full)

foreach (_t ${_benchmarks})
    add_executable(${_t} ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools Boost::mpi)
    #add_test(
    #    NAME ${_t}.cpp
    #    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_t} ${MPIEXEC_POSTFLAGS}
    #)
endforeach()
