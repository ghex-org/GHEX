# Variable used for benchmarks that DO NOT require multithreading support
set(_benchmarks ghex_p2p_bi_cb_avail ghex_p2p_bi_cb_wait ghex_p2p_bi_ft_avail ghex_p2p_bi_ft_wait mpi_p2p_bi_avail mpi_p2p_bi_wait)

# Variable used for benchmarks that require multithreading support
set(_benchmarks_mt ghex_p2p_bi_cb_avail ghex_p2p_bi_cb_wait ghex_p2p_bi_ft_avail ghex_p2p_bi_ft_wait mpi_p2p_bi_avail mpi_p2p_bi_wait)

foreach (_t ${_benchmarks})
    add_executable(${_t} ${_t}_mt.cpp )
    add_definitions(             -DUSE_HEAVY_CALLBACKS -DUSE_PMIX -DUSE_RAW_SHARED_MESSAGE -DUSE_POOL_ALLOCATOR)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIR} ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t})
endforeach()

foreach (_t ${_benchmarks_mt})
    add_executable(${_t}_mt ${_t}_mt.cpp )
    add_definitions(-DUSE_OPENMP -DUSE_HEAVY_CALLBACKS -DUSE_PMIX -DUSE_RAW_SHARED_MESSAGE -DUSE_POOL_ALLOCATOR)
    target_include_directories(${_t}_mt PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIR} ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t}_mt ${OpenMP_CXX_FLAGS})
    set_target_properties(${_t}_mt PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS} CXX_STANDARD 14)
endforeach()
