
add_subdirectory(mpi_runner)

set(serial_tests decomposition)
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
    ghex_reg_parallel_test(${t} 4 false)
endforeach()

foreach (t ${parallel_mt_tests})
    ghex_reg_parallel_test(${t} 4 true)
endforeach()

if (GHEX_USE_GPU)
    ghex_compile_test(unified_memory_allocator)
    ghex_reg_test(unified_memory_allocator)
endif()

add_subdirectory(structured)
add_subdirectory(unstructured)
add_subdirectory(glue)
add_subdirectory(bindings)
