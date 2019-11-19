
# simple benchmarks
# -----------------

# Variable used for benchmarks that DO NOT require multithreading support
set(_benchmarks_simple simple_comm_test_halo_exchange_3D_generic_full)
# Variable used for benchmarks that require multithreading support
set(_benchmarks_simple_mt )

foreach (_t ${_benchmarks_simple})
    add_executable(${_t} ${_t}.cpp )
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools gtest_main_bench)
endforeach()

foreach (_t ${_benchmarks_simple_mt})
    add_executable(${_t}_mt ${_t}.cpp )
    target_include_directories(${_t}_mt PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t}_mt MPI::MPI_CXX GridTools::gridtools gtest_main_bench_mt)
endforeach()


# advanced benchmarks
# -------------------

# Variable used for benchmarks that DO NOT require multithreading support
set(_benchmarks gcl_test_halo_exchange_3D_generic_full comm_2_test_halo_exchange_3D_generic_full)
# Variable used for benchmarks that require multithreading support
set(_benchmarks_mt )

foreach (_t ${_benchmarks})
    add_executable(${_t} ${_t}.cpp)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools gtest_main_bench)

    add_executable(${_t}_1_pattern ${_t}.cpp)
    target_include_directories(${_t}_1_pattern PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_compile_definitions(${_t}_1_pattern PUBLIC GHEX_1_PATTERN_BENCHMARK) 
    target_link_libraries(${_t}_1_pattern MPI::MPI_CXX GridTools::gridtools gtest_main_bench)

    if(USE_GPU)
        add_executable(${_t}_gpu ${_t}.cu)
        target_include_directories(${_t}_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_link_libraries(${_t}_gpu MPI::MPI_CXX GridTools::gridtools gtest_main_bench)

        add_executable(${_t}_1_pattern_gpu ${_t}.cu)
        target_include_directories(${_t}_1_pattern_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_compile_definitions(${_t}_1_pattern_gpu PUBLIC GHEX_1_PATTERN_BENCHMARK) 
        target_link_libraries(${_t}_1_pattern_gpu MPI::MPI_CXX GridTools::gridtools gtest_main_bench)
    endif()
endforeach()

foreach (_t ${_benchmarks_mt})
    add_executable(${_t}_mt ${_t}.cpp)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t}_mt MPI::MPI_CXX GridTools::gridtools gtest_main_bench_mt)

    add_executable(${_t}_1_pattern_mt ${_t}.cpp)
    target_include_directories(${_t}_1_pattern_mt PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_compile_definitions(${_t}_1_pattern_mt PUBLIC GHEX_1_PATTERN_BENCHMARK) 
    target_link_libraries(${_t}_1_pattern_mt MPI::MPI_CXX GridTools::gridtools gtest_main_bench_mt)

    if(USE_GPU)
        add_executable(${_t}_gpu_mt ${_t}.cu)
        target_include_directories(${_t}_gpu_mt PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_link_libraries(${_t}_gpu_mt MPI::MPI_CXX GridTools::gridtools gtest_main_bench_mt)

        add_executable(${_t}_1_pattern_gpu_mt ${_t}.cu)
        target_include_directories(${_t}_1_pattern_gpu_mt PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_compile_definitions(${_t}_1_pattern_gpu_mt PUBLIC GHEX_1_PATTERN_BENCHMARK) 
        target_link_libraries(${_t}_1_pattern_gpu_mt MPI::MPI_CXX GridTools::gridtools gtest_main_bench_mt)
    endif()
endforeach()


# atlas benchmarks
# ----------------

# Variable used for benchmarks that DO NOT require multithreading support
set(_benchmarks atlas_halo_exchange_nodecolumns)

foreach (_t ${_benchmarks})
    add_executable(${_t} ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX Boost::mpi GridTools::gridtools atlas)

    if(USE_GPU)
        add_executable(${_t}_gpu ${_t}.cu ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
        target_compile_definitions(${_t}_gpu PUBLIC GT_USE_GPU)
        target_include_directories(${_t}_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_link_libraries(${_t}_gpu MPI::MPI_CXX Boost::mpi GridTools::gridtools atlas)
    endif()
endforeach()
