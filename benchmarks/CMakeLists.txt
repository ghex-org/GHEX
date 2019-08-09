set(_benchmarks simple_comm_test_halo_exchange_3D_generic_full gcl_test_halo_exchange_3D_generic_full comm_2_test_halo_exchange_3D_generic_full)

foreach (_t ${_benchmarks})
    add_executable(${_t} ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools Boost::mpi)
endforeach()

add_executable(atlas_halo_exchange_nodecolumns atlas_halo_exchange_nodecolumns.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_include_directories(atlas_halo_exchange_nodecolumns PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(atlas_halo_exchange_nodecolumns MPI::MPI_CXX Boost::mpi GridTools::gridtools atlas)
