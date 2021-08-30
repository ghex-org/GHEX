function(ghex_target_compile_options target)
    target_compile_options(${target} PRIVATE
    # flags for CXX builds
    $<${c_cxx_lang}:$<BUILD_INTERFACE:-Wall -Wextra -Wpedantic -Wno-unknown-pragmas>>
    # flags for CUDA builds
    $<${cuda_lang}:$<BUILD_INTERFACE:-Xcompiler=-Wall -Wextra -Wpedantic -Wno-unknown-pragmas>>)
endfunction()

function(ghex_compile_test t_)
    set(t ${t_}_obj)
    add_library(${t} OBJECT test_${t_}.cpp)
    ghex_target_compile_options(${t})
    target_link_libraries(${t} PRIVATE GTest::gtest)
    target_link_libraries(${t} PUBLIC ghex)
endfunction()

function(ghex_reg_test t_)
    set(t ${t_})
    add_executable(${t} $<TARGET_OBJECTS:${t_}_obj>)
    ghex_target_compile_options(${t})
    target_link_libraries(${t} PRIVATE gtest_main)
    target_link_libraries(${t} PRIVATE ${LIBRT})
    add_test(
        NAME ${t}
        COMMAND $<TARGET_FILE:${t}>)
endfunction()

function(ghex_reg_parallel_test t_ lib n mt)
    if (${mt})
        set(t ${t_}_${lib}_mt)
    else()
        set(t ${t_}_${lib})
    endif()
    add_executable(${t} $<TARGET_OBJECTS:${t_}_obj>)
    ghex_target_compile_options(${t})
    if (${mt})
        target_link_libraries(${t} PRIVATE gtest_main_mpi_mt)
    else()
        target_link_libraries(${t} PRIVATE gtest_main_mpi)
    endif()
    target_link_libraries(${t} PRIVATE oomph::${lib})
    target_link_libraries(${t} PRIVATE ${LIBRT})
    add_test(
        NAME ${t}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${t}> ${MPIEXEC_POSTFLAGS})
endfunction()
