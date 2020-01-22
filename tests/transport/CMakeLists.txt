
#set(_tests test_low_level test_low_level_x test_send_multi test_cancel_request test_attach_detach)
set(_tests test_low_level)

foreach(t_ ${_tests})
    add_executable( ${t_} ./${t_}.cpp )
    target_link_libraries(${t_} gtest_main_mt)
    add_test(
        NAME ${t_}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${t_} ${MPIEXEC_POSTFLAGS}
    )

    if (GHEX_USE_UCP)
        add_executable( ${t_}_ucx ./${t_}.cpp )
        target_link_libraries(${t_}_ucx gtest_main_ucx)
        target_compile_definitions(${t_}_ucx PUBLIC GHEX_TEST_USE_UCX)
        add_test(
            NAME ${t_}_ucx
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${t_} ${MPIEXEC_POSTFLAGS}
        )
    endif()
endforeach(t_ ${_tests})

if (GHEX_USE_UCP)
    set(_tests_ucx test_ucx_context)

    foreach(t_ ${_tests_ucx})

        add_executable( ${t_} ./${t_}.cpp )
        target_link_libraries(${t_} gtest_main_ucx)
        add_test(
            NAME ${t_}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${t_} ${MPIEXEC_POSTFLAGS}
        )

    endforeach(t_ ${_tests_ucx})
endif()
