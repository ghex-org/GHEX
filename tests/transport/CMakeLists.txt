
set(_tests test_low_level test_low_level_x test_send_multi test_cancel_request test_attach_detach)

foreach(t_ ${_tests})

    add_executable( ${t_} ./${t_}.cpp )
    target_include_directories(${t_} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${t_}  GridTools::gridtools MPI::MPI_CXX gtest_main_mt)
    add_test(
        NAME ${t_}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${t_} ${MPIEXEC_POSTFLAGS}
    )

endforeach(t_ ${_tests})

set(_tests_ucx test_ucx_context)

foreach(t_ ${_tests_ucx})

    add_executable( ${t_} ./${t_}.cpp )
    target_include_directories(${t_} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS} ${UCP_INCLUDE_DIR})
    target_link_libraries(${t_}  GridTools::gridtools MPI::MPI_CXX gtest_main_ucx ${UCP_LIBRARY})
    add_test(
        NAME ${t_}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${t_} ${MPIEXEC_POSTFLAGS}
    )

endforeach(t_ ${_tests_ucx})
