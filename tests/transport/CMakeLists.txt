add_executable( test_low_level ./test_low_level.cpp )
target_link_libraries(test_low_level ${exe_LIBS}  ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)

add_executable( test_low_level_x ./test_low_level_x.cpp )
target_link_libraries(test_low_level_x ${exe_LIBS}  ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)
