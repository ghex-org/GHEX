
get_target_property(python_mod_path pyghex LIBRARY_OUTPUT_DIRECTORY)
get_filename_component(pyghex_test_workdir ${python_mod_path}/.. ABSOLUTE)

add_custom_target(pyghex_tests)
add_dependencies(pyghex pyghex_tests)
copy_files(TARGET pyghex_tests DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/fixtures FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/context.py)
copy_files(TARGET pyghex_tests DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/conftest.py)

function(ghex_reg_pytest t)
    copy_files(TARGET pyghex_tests DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/test_${t}.py)
    add_test(
        NAME py_${t}
        COMMAND ${PYTHON_EXECUTABLE} -m pytest -s ${CMAKE_CURRENT_BINARY_DIR}/test_${t}.py
        WORKING_DIRECTORY ${pyghex_test_workdir})
    set_tests_properties(py_${t} PROPERTIES RUN_SERIAL ON)
endfunction()

function(ghex_reg_parallel_pytest t n)
    copy_files(TARGET pyghex_tests DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/test_${t}.py)
    add_test(
        NAME py_${t}_parallel
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${MPIEXEC_PREFLAGS}
        ${PYTHON_EXECUTABLE} -m pytest -s --with-mpi ${CMAKE_CURRENT_BINARY_DIR}/test_${t}.py
        WORKING_DIRECTORY ${pyghex_test_workdir})
    set_tests_properties(py_${t}_parallel PROPERTIES RUN_SERIAL ON)
endfunction()

ghex_reg_pytest(context)
ghex_reg_parallel_pytest(context 4)

ghex_reg_parallel_pytest(structured_domain_descriptor 4)
ghex_reg_parallel_pytest(structured_pattern 4)

#ghex_reg_pytest(unstructured_domain_descriptor)
ghex_reg_parallel_pytest(unstructured_domain_descriptor 4)
