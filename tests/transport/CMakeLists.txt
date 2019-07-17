
set(_tests test_low_level test_low_level_x test_send_multi test_cancel_request)



foreach(t_ ${_tests})

    add_executable( ${t_} ./${t_}.cpp )
    target_include_directories(${t_} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${t_}  GridTools::gridtools MPI::MPI_CXX Boost::mpi gtest_main_mt)
    add_test(
        NAME ${t_}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${t_} ${MPIEXEC_POSTFLAGS}
    )

endforeach(t_ ${_tests})
# add_executable( test_low_level_x ./test_low_level_x.cpp  ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
# target_include_directories(test_low_level_x PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
# target_link_libraries(test_low_level_x  GridTools::gridtools MPI::MPI_CXX Boost::mpi)

# add_executable( test_send_multi ./test_send_multi.cpp  ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
# target_include_directories(test_send_multi PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
# target_link_libraries(test_send_multi  GridTools::gridtools MPI::MPI_CXX Boost::mpi)

# add_executable( test_cancel_request ./test_cancel_request.cpp  ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
# target_include_directories(test_cancel_request PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
# target_link_libraries(test_cancel_request  GridTools::gridtools MPI::MPI_CXX Boost::mpi)

