
if (GHEX_FORTRAN_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

function(ghex_reg_parallel_test_f t n)
    add_executable(${t} test_${t}.f90)
    set_target_properties(${t} PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${t}_Mod)
    ghex_target_compile_options(${t})
    target_link_libraries(${t} PRIVATE fhex)
    if (GHEX_FORTRAN_OPENMP)
        target_link_libraries(${t} PRIVATE OpenMP::OpenMP_Fortran)
    endif()
    ghex_link_to_oomph(${t})
    ## workaround for clang+openmp
    #target_link_libraries(${t} PRIVATE $<$<CXX_COMPILER_ID:Clang>:$<LINK_ONLY:-fopenmp=libomp>>)
    add_test(
        NAME ${t}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${t}> ${MPIEXEC_POSTFLAGS})
endfunction()

ghex_reg_parallel_test_f(f_context 4)
if (GHEX_FORTRAN_OPENMP AND OOMPH_ENABLE_BARRIER)
    ghex_reg_parallel_test_f(f_barrier 4)
endif()
ghex_reg_parallel_test_f(f_structured 4)
ghex_reg_parallel_test_f(f_unstructured 4)
