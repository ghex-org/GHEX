
include(ghex_compile_options)

function(ghex_reg_parallel_test_f t n)
    add_executable(${t} test_${t}.f90)
    ghex_target_compile_options(${t})
    target_link_libraries(${t} PRIVATE fhex)
    ## workaround for clang+openmp
    #target_link_libraries(${t} PRIVATE $<$<CXX_COMPILER_ID:Clang>:$<LINK_ONLY:-fopenmp=libomp>>)
    add_test(
        NAME ${t}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${t}> ${MPIEXEC_POSTFLAGS})
endfunction()

ghex_reg_parallel_test_f(f_context 4)
