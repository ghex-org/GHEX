
include(ghex_compile_options)

function(ghex_reg_parallel_test_f t n)
    add_executable(${t} test_${t}.f90)
    ghex_target_compile_options(${t})
    target_link_libraries(${t} PRIVATE fhex)
    add_test(
        NAME ${t}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${t}> ${MPIEXEC_POSTFLAGS})
endfunction()

ghex_reg_parallel_test_f(f_context 4)
