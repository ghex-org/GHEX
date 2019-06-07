set(_tests communication_object)

foreach (_t ${_tests})
    add_executable(${_t} ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools Boost::mpi)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_t} ${MPIEXEC_POSTFLAGS}
    )
endforeach()
