# simple benchmarks
# -----------------

# Variable used for benchmarks that DO NOT require multithreading support
set(_benchmarks_simple simple_comm_test_halo_exchange_3D_generic_full)
# Variable used for benchmarks that require multithreading support
set(_benchmarks_simple_mt )
foreach (_t ${_benchmarks_simple})
    add_executable(${_t} ${_t}.cpp )
    target_link_libraries(${_t} gtest_main_bench)
endforeach()

foreach (_t ${_benchmarks_simple_mt})
    add_executable(${_t}_mt ${_t}.cpp )
    target_link_libraries(${_t}_mt gtest_main_bench_mt)
endforeach()


# advanced benchmarks
# -------------------

# Variable used for benchmarks that DO NOT require multithreading support
set(_benchmarks gcl_test_halo_exchange_3D_generic_full comm_2_test_halo_exchange_3D_generic_full)
# Variable used for benchmarks that require multithreading support
set(_benchmarks_mt )

foreach (_t ${_benchmarks})
    add_executable(${_t} ${_t}.cpp)
    target_link_libraries(${_t} gtest_main_bench)

    add_executable(${_t}_1_pattern ${_t}.cpp)
    target_compile_definitions(${_t}_1_pattern PUBLIC GHEX_1_PATTERN_BENCHMARK GCL_MPI) 
    target_link_libraries(${_t}_1_pattern gtest_main_bench)

    if(USE_GPU)
        add_executable(${_t}_gpu ${_t}.cu)
        target_link_libraries(${_t}_gpu gtest_main_bench)

        add_executable(${_t}_1_pattern_gpu ${_t}.cu)
        target_compile_definitions(${_t}_1_pattern_gpu PUBLIC GHEX_1_PATTERN_BENCHMARK)
        target_link_libraries(${_t}_1_pattern_gpu gtest_main_bench)
    endif()
endforeach()

foreach (_t ${_benchmarks_mt})
    add_executable(${_t}_mt ${_t}.cpp)
    target_link_libraries(${_t}_mt gtest_main_bench_mt)

    add_executable(${_t}_1_pattern_mt ${_t}.cpp)
    target_compile_definitions(${_t}_1_pattern_mt PUBLIC GHEX_1_PATTERN_BENCHMARK)
    target_link_libraries(${_t}_1_pattern_mt gtest_main_bench_mt)

    if(USE_GPU)
        add_executable(${_t}_gpu_mt ${_t}.cu)
        target_link_libraries(${_t}_gpu_mt gtest_main_bench_mt)

        add_executable(${_t}_1_pattern_gpu_mt ${_t}.cu)
        target_compile_definitions(${_t}_1_pattern_gpu_mt PUBLIC GHEX_1_PATTERN_BENCHMARK)
        target_link_libraries(${_t}_1_pattern_gpu_mt gtest_main_bench_mt)
    endif()
endforeach()

add_subdirectory(transport)


# atlas benchmarks
# ----------------

if (GHEX_ENABLE_ATLAS_BINDINGS)
    # Variable used for benchmarks that DO NOT require multithreading support
    set(_benchmarks atlas_halo_exchange_nodecolumns)

    foreach (_t ${_benchmarks})
        add_executable(${_t} ${_t}.cpp)
        target_link_libraries(${_t} gtest_main_bench_atlas)

        if(USE_GPU)
            add_executable(${_t}_gpu ${_t}.cu)
            target_compile_definitions(${_t}_gpu PUBLIC GT_USE_GPU)
            target_link_libraries(${_t}_gpu gtest_main_bench_atlas)
        endif()
    endforeach()
endif()
