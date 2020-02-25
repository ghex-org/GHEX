set(_serial_tests aligned_allocator unified_memory_allocator)
foreach (_t ${_serial_tests})
    add_executable(${_t} ${_t}.cpp)
    target_link_libraries(${_t} gtest_main_mt)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
    )
endforeach()

if (USE_GPU)
    set(_serial_tests_gpu unified_memory_allocator)
    foreach (_t ${_serial_tests_gpu})
        add_executable(${_t}_gpu ${_t}.cu)
        target_compile_definitions(${_t}_gpu PUBLIC GT_USE_GPU)
        target_link_libraries(${_t}_gpu gtest_main_mt)
        add_test(
            NAME ${_t}.cu
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
        )
    endforeach()
endif()

#set(_tests mpi_allgather communication_object)
set(_tests mpi_allgather)

foreach (_t ${_tests})
    add_executable(${_t} ${_t}.cpp)
    target_link_libraries(${_t} gtest_main_mt)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
    )
endforeach()


if (GHEX_USE_UCP)
    set(_variants_ucx serial serial_split threads async_async async_deferred)

    foreach(_var ${_variants_ucx})
        string(TOUPPER ${_var} define)
        set(_t communication_object_2_${_var}_ucx)
        add_executable(${_t} communication_object_2.cpp )
        target_compile_definitions(${_t} PUBLIC GHEX_TEST_${define} GHEX_TEST_USE_UCX)
        target_link_libraries(${_t} gtest_main_ucx)
        add_test(
            NAME ${_t}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
        )

        set(_t communication_object_2_${_var}_vector_ucx)
        add_executable(${_t} communication_object_2.cpp )
        target_compile_definitions(${_t} PUBLIC GHEX_TEST_${define}_VECTOR GHEX_TEST_USE_UCX)
        target_link_libraries(${_t} gtest_main_ucx)
        add_test(
            NAME ${_t}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
        )
    endforeach()
endif()


set(_variants serial serial_split threads async_async async_deferred)
foreach(_var ${_variants})
    string(TOUPPER ${_var} define)
    set(_t communication_object_2_${_var})
    add_executable(${_t} communication_object_2.cpp)
    target_compile_definitions(${_t} PUBLIC GHEX_TEST_${define})
    target_link_libraries(${_t} gtest_main_mt)
    add_test(
        NAME ${_t}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
    )

    set(_t communication_object_2_${_var}_vector)
    add_executable(${_t} communication_object_2.cpp)
    target_compile_definitions(${_t} PUBLIC GHEX_TEST_${define}_VECTOR)
    target_link_libraries(${_t} gtest_main_mt)
    add_test(
        NAME ${_t}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
    )

    if (USE_HYBRID_TESTS)
        set(_t communication_object_2_${_var}_hybrid)
        add_executable(${_t} communication_object_2.cpp )
        target_compile_definitions(${_t} PUBLIC GHEX_TEST_${define})
        target_compile_definitions(${_t} PUBLIC GHEX_EMULATE_GPU)
        target_compile_definitions(${_t} PUBLIC GHEX_HYBRID_TESTS)
        target_link_libraries(${_t} gtest_main_mt)
        add_test(
            NAME ${_t}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
        )
    endif()

    if (USE_GPU)
        set(_t communication_object_2_${_var}_gpu)
        add_executable(${_t} communication_object_2.cu)
        target_compile_definitions(${_t} PUBLIC GHEX_TEST_${define})
        target_link_libraries(${_t} gtest_main_mt)
        add_test(
            NAME ${_t}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
        )

        set(_t communication_object_2_${_var}_vector_gpu)
        add_executable(${_t} communication_object_2.cu)
        target_compile_definitions(${_t} PUBLIC GHEX_TEST_${define}_VECTOR)
        target_link_libraries(${_t} gtest_main_mt)
        add_test(
            NAME ${_t}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
        )

        if (USE_HYBRID_TESTS)
            set(_t communication_object_2_${_var}_hybrid_gpu)
            add_executable(${_t} communication_object_2.cu)
            target_compile_definitions(${_t} PUBLIC GHEX_TEST_${define})
            target_compile_definitions(${_t} PUBLIC GHEX_HYBRID_TESTS)
            target_link_libraries(${_t} gtest_main_mt)
            add_test(
                NAME ${_t}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
            )
        endif()
    endif()
endforeach(_var)

set(_tests_gt data_store_test)
foreach (_t ${_tests_gt})
    add_executable(${_t} ${_t}.cpp)
    target_link_libraries(${_t} gtest_main_mt)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
    )
endforeach()

if (GHEX_ENABLE_ATLAS_BINDINGS)
    set(_tests_atlas atlas_integration atlas_integration_comm_2)
    foreach (_t ${_tests_atlas})
        add_executable(${_t} ${_t}.cpp)
        target_link_libraries(${_t} gtest_main_atlas)
        add_test(
            NAME ${_t}.cpp
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
            )
    endforeach()
endif()

if (USE_GPU)
    foreach (_t ${_tests_gt})
        add_executable(${_t}_gpu ${_t}.cu)
        target_compile_definitions(${_t}_gpu PUBLIC GT_USE_GPU)
        target_link_libraries(${_t}_gpu gtest_main_mt)
        add_test(
            NAME ${_t}.cu
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
        )
    endforeach()

    if (GHEX_ENABLE_ATLAS_BINDINGS)
        set(_tests_atlas_gpu atlas_integration_comm_2)
        foreach (_t ${_tests_atlas_gpu})
            add_executable(${_t}_gpu ${_t}.cu)
            target_compile_definitions(${_t}_gpu PUBLIC GT_USE_GPU)
            target_link_libraries(${_t}_gpu gtest_main_atlas)
            add_test(
                NAME ${_t}.cu
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
            )
        endforeach()
    endif()
endif()

add_subdirectory(transport)
