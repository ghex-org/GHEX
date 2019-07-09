add_executable( test_low_level ./test_low_level.cpp )
target_link_libraries(test_low_level ${exe_LIBS}  ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)

add_executable( test_low_level_x ./test_low_level_x.cpp )
target_link_libraries(test_low_level_x ${exe_LIBS}  ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)

add_executable( test_send_multi ./test_send_multi.cpp )
target_link_libraries(test_send_multi ${exe_LIBS}  ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)

add_executable( test_cancel_request ./test_cancel_request.cpp )
target_link_libraries(test_cancel_request ${exe_LIBS}  ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)
