#set(_benchmarks simple_comm_test_halo_exchange_3D_generic_full gcl_test_halo_exchange_3D_generic_full comm_2_test_halo_exchange_3D_generic_full)
set(_benchmarks simple_comm_test_halo_exchange_3D_generic_full)

foreach (_t ${_benchmarks})
    add_executable(${_t} ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools Boost::mpi)
endforeach()

set(_benchmarks_2 gcl_test_halo_exchange_3D_generic_full comm_2_test_halo_exchange_3D_generic_full)

foreach (_t ${_benchmarks_2})
    add_executable(${_t} ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools Boost::mpi)

    add_executable(${_t}_1_pattern ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t}_1_pattern PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_compile_definitions(${_t}_1_pattern PUBLIC GHEX_1_PATTERN_BENCHMARK) 
    target_link_libraries(${_t}_1_pattern MPI::MPI_CXX GridTools::gridtools Boost::mpi)

    if(USE_GPU)
        add_executable(${_t}_gpu ${_t}.cu ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
        target_include_directories(${_t}_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_link_libraries(${_t}_gpu MPI::MPI_CXX GridTools::gridtools Boost::mpi)

        add_executable(${_t}_1_pattern_gpu ${_t}.cu ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
        target_include_directories(${_t}_1_pattern_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_compile_definitions(${_t}_1_pattern_gpu PUBLIC GHEX_1_PATTERN_BENCHMARK) 
        target_link_libraries(${_t}_1_pattern_gpu MPI::MPI_CXX GridTools::gridtools Boost::mpi)
    endif()
endforeach()
