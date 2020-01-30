# Variable used for benchmarks that DO NOT require multithreading support
set(_benchmarks ghex_p2p_bi_cb_avail ghex_p2p_bi_cb_wait ghex_p2p_bi_ft_avail ghex_p2p_bi_ft_wait)

# Variable used for benchmarks that require multithreading support
set(_benchmarks_mt ghex_p2p_bi_cb_avail ghex_p2p_bi_cb_wait ghex_p2p_bi_ft_avail ghex_p2p_bi_ft_wait)

foreach (_t ${_benchmarks})
    add_executable(${_t} ${_t}_mt.cpp )
    target_compile_definitions(${_t} PRIVATE USE_HEAVY_CALLBACKS USE_RAW_SHARED_MESSAGE USE_POOL_ALLOCATOR)
    target_link_libraries(${_t} ghexlib)
endforeach()

foreach (_t ${_benchmarks_mt})
    add_executable(${_t}_mt ${_t}_mt.cpp )
    target_compile_definitions(${_t}_mt PRIVATE USE_OPENMP USE_HEAVY_CALLBACKS USE_RAW_SHARED_MESSAGE USE_POOL_ALLOCATOR)
    target_link_libraries(${_t}_mt ghexlib)
endforeach()

if (GHEX_USE_UCP)
   foreach (_t ${_benchmarks})
        add_executable(${_t}_ucx ${_t}_mt.cpp )
        target_compile_definitions(${_t}_ucx PRIVATE USE_HEAVY_CALLBACKS USE_RAW_SHARED_MESSAGE USE_POOL_ALLOCATOR)
        target_link_libraries(${_t}_ucx ghexlib)
        target_compile_definitions(${_t}_ucx PRIVATE USE_UCP)
        if (GHEX_USE_PMIX)
            target_compile_definitions(${_t}_ucx PRIVATE GHEX_USE_PMI)
        endif()
    endforeach()

    foreach (_t ${_benchmarks_mt})
        add_executable(${_t}_mt_ucx ${_t}_mt.cpp )
        target_compile_definitions(${_t}_mt_ucx PRIVATE USE_OPENMP USE_HEAVY_CALLBACKS USE_RAW_SHARED_MESSAGE USE_POOL_ALLOCATOR)
        target_link_libraries(${_t}_mt_ucx ghexlib)
        target_compile_definitions(${_t}_mt_ucx PRIVATE USE_UCP)
        if (GHEX_USE_PMIX)
            target_compile_definitions(${_t}_mt_ucx PRIVATE GHEX_USE_PMI)
        endif()
    endforeach()
endif()
