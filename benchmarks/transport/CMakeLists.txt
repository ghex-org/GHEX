# Variable used for benchmarks that DO NOT require multithreading support
set(_benchmarks ghex_p2p_bi_cb_avail ghex_p2p_bi_cb_wait ghex_p2p_bi_ft_avail ghex_p2p_bi_ft_wait)

# Variable used for benchmarks that require multithreading support
set(_benchmarks_mt ghex_p2p_bi_cb_avail ghex_p2p_bi_cb_wait ghex_p2p_bi_ft_avail ghex_p2p_bi_ft_wait)

foreach (_t ${_benchmarks})
    add_executable(${_t} ${_t}_mt.cpp )
    target_link_libraries(${_t} ghexlib)
endforeach()

if (OpenMP_FOUND)
    foreach (_t ${_benchmarks_mt})
        add_executable(${_t}_mt ${_t}_mt.cpp )
        target_compile_definitions(${_t}_mt PRIVATE GHEX_USE_OPENMP)
        target_link_libraries(${_t}_mt ghexlib OpenMP::OpenMP_CXX)
    endforeach()
endif()

if (GHEX_USE_UCP)
    find_package(Threads REQUIRED)
    foreach (_t ${_benchmarks})
        add_executable(${_t}_ucx ${_t}_mt.cpp )
        target_compile_definitions(${_t}_ucx PRIVATE GHEX_USE_UCP)
        target_link_libraries(${_t}_ucx ghexlib Threads::Threads)
    endforeach()

    if (OpenMP_FOUND)
        foreach (_t ${_benchmarks_mt})
            add_executable(${_t}_mt_ucx ${_t}_mt.cpp )
            target_compile_definitions(${_t}_mt_ucx PRIVATE GHEX_USE_OPENMP GHEX_USE_UCP)
            target_link_libraries(${_t}_mt_ucx ghexlib OpenMP::OpenMP_CXX)
        endforeach()
    endif()
endif()

if (GHEX_BUILD_FORTRAN)
    enable_language(Fortran)
    set(_benchmarks_f fhex_p2p_bi_cb_avail fhex_p2p_bi_cb_wait fhex_p2p_bi_ft_avail fhex_p2p_bi_ft_wait)
    set(_benchmarks_f_mt fhex_p2p_bi_cb_avail fhex_p2p_bi_cb_wait fhex_p2p_bi_ft_avail fhex_p2p_bi_ft_wait)

    foreach (_t ${_benchmarks_f})
        add_executable(${_t} ${_t}_mt.f90 )
        target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/bindings/ )
        target_link_libraries(${_t} ghexlib fhex)
    endforeach()

    foreach (_t ${_benchmarks_f_mt})
        add_executable(${_t}_mt ${_t}_mt.f90 )
        target_compile_definitions(${_t}_mt PRIVATE GHEX_USE_OPENMP)
        target_include_directories(${_t}_mt PRIVATE ${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/bindings/ )
        target_link_libraries(${_t}_mt ghexlib fhex OpenMP::OpenMP_Fortran)
    endforeach()
endif()
