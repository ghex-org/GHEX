set(_serial_cubed_sphere_tests transform)
foreach (_t ${_serial_cubed_sphere_tests})
    set(_name cubed_sphere_${_t})
    add_executable(${_name} ${_t}.cpp)
    target_link_libraries(${_name} gtest_main_mt)
    add_test(
        NAME ${_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_name}> ${MPIEXEC_POSTFLAGS}
    )
endforeach()

set(_cubed_sphere_tests exchange)
# MPI, CPU
foreach (_x ${_cubed_sphere_tests})
    set(_t cubed_sphere_${_x})
    add_executable(${_t} ${_x}.cpp)
    target_link_libraries(${_t} gtest_main_mt)
    add_test(
        NAME ${_t}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
    )
endforeach()
# MPI, GPU
if (USE_GPU)
    foreach (_x ${_cubed_sphere_tests})
        set(_t cubed_sphere_${_x}_gpu)
        add_ghex_gpu_executable(${_t} ${_x})
        target_link_libraries(${_t} gtest_main_mt)
        add_test(
            NAME ${_t}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
        )
    endforeach()
endif()
# UCX, CPU
if (GHEX_USE_UCP)
    foreach (_x ${_cubed_sphere_tests})
        set(_t cubed_sphere_${_x}_ucx)
        add_executable(${_t} ${_x}.cpp)
        target_compile_definitions(${_t} PUBLIC GHEX_TEST_USE_UCX)
        target_link_libraries(${_t} gtest_main_mt)
        add_test(
            NAME ${_t}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
        )
    endforeach()
    # UCX, GPU
    if (USE_GPU)
        foreach (_x ${_simple_tests})
            set(_t cubed_sphere_${_x}_ucx_gpu)
            add_ghex_gpu_executable(${_t} ${_x})
            target_compile_definitions(${_t} PUBLIC GHEX_TEST_USE_UCX)
            target_link_libraries(${_t} gtest_main_mt)
            add_test(
                NAME ${_t}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
            )
        endforeach()
    endif()
endif()
