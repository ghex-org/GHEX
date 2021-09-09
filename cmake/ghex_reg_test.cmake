include(ghex_compile_options)

function(ghex_compile_test t_)
    set(t ${t_}_obj)
    compile_as_cuda(test_${t_}.cpp)
    add_library(${t} OBJECT test_${t_}.cpp)
    ghex_target_compile_options(${t})
    link_device_runtime(${t})
    target_link_libraries(${t} PRIVATE GTest::gtest)
    target_link_libraries(${t} PUBLIC ghex_common)
endfunction()

function(ghex_reg_test t_)
    set(t ${t_})
    add_executable(${t} $<TARGET_OBJECTS:${t_}_obj>)
    target_link_libraries(${t} PRIVATE gtest_main)
    target_link_libraries(${t} PRIVATE ghex)
    # workaround for clang+openmp
    target_link_libraries(${t} PRIVATE $<$<CXX_COMPILER_ID:Clang>:$<LINK_ONLY:-fopenmp=libomp>>)
    add_test(
        NAME ${t}
        COMMAND $<TARGET_FILE:${t}>)
endfunction()

function(ghex_reg_parallel_test t_ n mt)
    if (${mt})
        set(t ${t_}_mt)
        add_executable(${t} $<TARGET_OBJECTS:${t_}_obj>)
        target_link_libraries(${t} PRIVATE gtest_main_mpi_mt)
    else()
        set(t ${t_})
        add_executable(${t} $<TARGET_OBJECTS:${t_}_obj>)
        target_link_libraries(${t} PRIVATE gtest_main_mpi)
    endif()
    target_link_libraries(${t} PRIVATE ghex)
    # workaround for clang+openmp
    target_link_libraries(${t} PRIVATE $<$<CXX_COMPILER_ID:Clang>:$<LINK_ONLY:-fopenmp=libomp>>)
    add_test(
        NAME ${t}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${t}> ${MPIEXEC_POSTFLAGS})
endfunction()
