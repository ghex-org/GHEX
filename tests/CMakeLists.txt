set(_tests mpi_allgather communication_object)

foreach (_t ${_tests})
    add_executable(${_t} ${_t}.cpp)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools Boost::mpi gtest_main_mt )
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_t} ${MPIEXEC_POSTFLAGS}
    )
endforeach()

set(_variants serial serial_split threads async_async async_deferred async_async_wait)

foreach(_var ${_variants})
    string(TOUPPER ${_var} define)
    add_executable(communication_object_2_${_var} communication_object_2.cpp )
    target_compile_definitions(communication_object_2_${_var} PUBLIC GHEX_TEST_${define})
    target_include_directories(communication_object_2_${_var} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(communication_object_2_${_var} MPI::MPI_CXX GridTools::gridtools Boost::mpi gtest_main_mt )
    add_test(
        NAME communication_object_2_${_var}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} communication_object_2_${_var} ${MPIEXEC_POSTFLAGS}
    )
endforeach(_var)

add_subdirectory(transport)