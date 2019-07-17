set(_benchmarks simple_comm_test_halo_exchange_3D_generic_full gcl_test_halo_exchange_3D_generic_full comm_2_test_halo_exchange_3D_generic_full)

foreach (_t ${_benchmarks})
    add_executable(${_t} ${_t}.cpp )
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools Boost::mpi gtest_main_bench)
endforeach()

foreach (_t ${_benchmarks})
    add_executable(${_t}_mt ${_t}.cpp )
    target_include_directories(${_t}_mt PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t}_mt MPI::MPI_CXX GridTools::gridtools Boost::mpi gtest_main_bench_mt)
endforeach()
