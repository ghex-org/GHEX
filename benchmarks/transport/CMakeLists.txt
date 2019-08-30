# Variable used for benchmarks that DO NOT require multithreading support
set(_benchmarks )

# Variable used for benchmarks that require multithreading support
set(_benchmarks_mt mpi_thread_multiple)

foreach (_t ${_benchmarks_mt})
    add_executable(${_t}_mt ${_t}.cpp )
    target_include_directories(${_t}_mt PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t}_mt MPI::MPI_CXX GridTools::gridtools Boost::mpi gtest_main_bench_mt)
endforeach()
