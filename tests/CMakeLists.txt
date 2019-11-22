
if(USE_MPI_WITH_UCX_IN_TESTS)
    set(_ucx_params -mca pml ucx)
else()
    set(_ucx_params )
endif()

set(_serial_tests aligned_allocator)

foreach (_t ${_serial_tests})
    add_executable(${_t} ${_t}.cpp)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools gtest_main_mt)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${_ucx_params} ${_t} ${MPIEXEC_POSTFLAGS}
    )
endforeach()

set(_tests mpi_allgather communication_object)

foreach (_t ${_tests})
    add_executable(${_t} ${_t}.cpp)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools gtest_main_mt)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} ${_t} ${MPIEXEC_POSTFLAGS}
    )
endforeach()

set(_variants serial serial_split threads async_async async_deferred async_async_wait)

foreach(_var ${_variants})
    string(TOUPPER ${_var} define)

    add_executable(communication_object_2_${_var} communication_object_2.cpp )
    target_compile_definitions(communication_object_2_${_var} PUBLIC GHEX_TEST_${define})
    target_include_directories(communication_object_2_${_var} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(communication_object_2_${_var} MPI::MPI_CXX GridTools::gridtools gtest_main_mt)
    add_test(
        NAME communication_object_2_${_var}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_${_var} ${MPIEXEC_POSTFLAGS}
    )

    add_executable(communication_object_2_${_var}_vector communication_object_2.cpp )
    target_compile_definitions(communication_object_2_${_var}_vector PUBLIC GHEX_TEST_${define}_VECTOR)
    target_include_directories(communication_object_2_${_var}_vector PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(communication_object_2_${_var}_vector MPI::MPI_CXX GridTools::gridtools gtest_main_mt)
    add_test(
        NAME communication_object_2_${_var}_vector
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_${_var}_vector ${MPIEXEC_POSTFLAGS}
    )

    if (USE_HYBRID_TESTS)
        add_executable(communication_object_2_${_var}_hybrid communication_object_2.cpp )
        target_compile_definitions(communication_object_2_${_var}_hybrid PUBLIC GHEX_TEST_${define})
        target_compile_definitions(communication_object_2_${_var}_hybrid PUBLIC GHEX_EMULATE_GPU)
        target_compile_definitions(communication_object_2_${_var}_hybrid PUBLIC GHEX_HYBRID_TESTS)
        target_include_directories(communication_object_2_${_var}_hybrid PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_link_libraries(communication_object_2_${_var}_hybrid MPI::MPI_CXX GridTools::gridtools gtest_main_mt)
        add_test(
            NAME communication_object_2_${_var}_hybrid
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_${_var}_hybrid ${MPIEXEC_POSTFLAGS}
        )
    endif()

    if (USE_GPU)
        add_executable(communication_object_2_${_var}_gpu communication_object_2.cu)
        target_compile_definitions(communication_object_2_${_var}_gpu PUBLIC GHEX_TEST_${define})
        target_include_directories(communication_object_2_${_var}_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_link_libraries(communication_object_2_${_var}_gpu MPI::MPI_CXX GridTools::gridtools gtest_main_mt)
        add_test(
            NAME communication_object_2_${_var}_gpu
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_${_var}_gpu ${MPIEXEC_POSTFLAGS}
        )

        add_executable(communication_object_2_${_var}_vector_gpu communication_object_2.cu)
        target_compile_definitions(communication_object_2_${_var}_vector_gpu PUBLIC GHEX_TEST_${define}_VECTOR)
        target_include_directories(communication_object_2_${_var}_vector_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_link_libraries(communication_object_2_${_var}_vector_gpu MPI::MPI_CXX GridTools::gridtools gtest_main_mt)
        add_test(
            NAME communication_object_2_${_var}_vector_gpu
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_${_var}_vector_gpu ${MPIEXEC_POSTFLAGS}
        )

        if (USE_HYBRID_TESTS)
            add_executable(communication_object_2_${_var}_hybrid_gpu communication_object_2.cu)
            target_compile_definitions(communication_object_2_${_var}_hybrid_gpu PUBLIC GHEX_TEST_${define})
            target_compile_definitions(communication_object_2_${_var}_hybrid_gpu PUBLIC GHEX_HYBRID_TESTS)
            target_include_directories(communication_object_2_${_var}_hybrid_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
            target_link_libraries(communication_object_2_${_var}_hybrid_gpu MPI::MPI_CXX GridTools::gridtools gtest_main_mt)
            add_test(
                NAME communication_object_2_${_var}_hybrid_gpu
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_${_var}_hybrid_gpu ${MPIEXEC_POSTFLAGS}
            )
        endif()
    endif()
endforeach(_var)

set(_tests_gt data_store_test)

foreach (_t ${_tests_gt})
    add_executable(${_t} ${_t}.cpp)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools gtest_main_mt)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} ${_t} ${MPIEXEC_POSTFLAGS}
    )
endforeach()

set(_tests_atlas atlas_integration atlas_integration_comm_2)

foreach (_t ${_tests_atlas})
    add_executable(${_t} ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools atlas)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_t} ${MPIEXEC_POSTFLAGS}
    )
endforeach()

if (USE_GPU)
    foreach (_t ${_tests_gt})
        add_executable(${_t}_gpu ${_t}.cu)
        target_compile_definitions(${_t}_gpu PUBLIC GT_USE_GPU)
        target_include_directories(${_t}_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_link_libraries(${_t}_gpu MPI::MPI_CXX GridTools::gridtools gtest_main_mt)
        add_test(
            NAME ${_t}.cu
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} ${_t} ${MPIEXEC_POSTFLAGS}
        )
    endforeach()

    set(_tests_atlas_gpu atlas_integration_comm_2 atlas_integration_comm_2_GHEX_only)

    foreach (_t ${_tests_atlas_gpu})
        add_executable(${_t}_gpu ${_t}.cu ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
        target_compile_definitions(${_t}_gpu PUBLIC GT_USE_GPU)
        target_include_directories(${_t}_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_link_libraries(${_t}_gpu MPI::MPI_CXX GridTools::gridtools atlas)
    endforeach()
endif()

add_subdirectory(transport)
