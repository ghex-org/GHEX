if (GHEX_USE_UCP)
    # Variable used for benchmarks that DO NOT require multithreading support
    #set(_benchmarks ghex_p2p_bi_cb_avail ghex_p2p_bi_cb_wait ghex_p2p_bi_ft_avail ghex_p2p_bi_ft_wait mpi_p2p_bi_avail mpi_p2p_bi_wait)
    set(_benchmarks ghex_p2p_bi_cb_avail)

    # Variable used for benchmarks that require multithreading support
    #set(_benchmarks_mt ghex_p2p_bi_cb_avail ghex_p2p_bi_cb_wait ghex_p2p_bi_ft_avail ghex_p2p_bi_ft_wait mpi_p2p_bi_avail mpi_p2p_bi_wait)
    set(_benchmarks_mt ghex_p2p_bi_cb_avail)

   foreach (_t ${_benchmarks})
        add_executable(${_t} ${_t}_mt.cpp )
        target_compile_definitions(${_t} PRIVATE USE_HEAVY_CALLBACKS USE_RAW_SHARED_MESSAGE USE_POOL_ALLOCATOR)
        target_link_libraries(${_t} ghexlib)
        if (GHEX_USE_PMIX)
            target_compile_definitions(${_t} PRIVATE USE_PMIX)
        endif()
    endforeach()

    foreach (_t ${_benchmarks_mt})
        add_executable(${_t}_mt ${_t}_mt.cpp )
        target_compile_definitions(${_t}_mt PRIVATE USE_OPENMP USE_HEAVY_CALLBACKS USE_RAW_SHARED_MESSAGE USE_POOL_ALLOCATOR)
        target_link_libraries(${_t}_mt ghexlib)
        if (GHEX_USE_PMIX)
            target_compile_definitions(${_t}_mt PRIVATE USE_PMIX)
        endif()
    endforeach()
endif()
