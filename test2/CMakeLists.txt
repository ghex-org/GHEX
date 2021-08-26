
include(ghex_gtest)
include(ghex_reg_test)

add_subdirectory(mpi_runner)

set(serial_tests aligned_allocator)
set(parallel_tests context mpi_communicator)
set(parallel_mt_tests context)

list(APPEND all_tests ${serial_tests} ${parallel_tests} ${parallel_mt_tests})
list(REMOVE_DUPLICATES all_tests)

foreach (t ${all_tests})
    ghex_compile_test(${t})
endforeach()

foreach (t ${serial_tests})
    ghex_reg_test(${t})
endforeach()

foreach (t ${parallel_tests})
    ghex_reg_parallel_test(${t} mpi 4 false)
    ghex_reg_parallel_test(${t} ucx 4 false)
endforeach()

foreach (t ${parallel_mt_tests})
    ghex_reg_parallel_test(${t} mpi 4 true)
    ghex_reg_parallel_test(${t} ucx 4 true)
endforeach()

add_subdirectory(structured)
