# paths
# =====

# virtual environment
set(venv_dir "${CMAKE_BINARY_DIR}/test_venv")
# test requirements
set(reqs "${CMAKE_SOURCE_DIR}/bindings/python/min-requirements-test.txt")
# get the path to the pyghex module's parent directory
get_target_property(python_mod_path pyghex LIBRARY_OUTPUT_DIRECTORY)
get_filename_component(pyghex_test_workdir ${python_mod_path}/.. ABSOLUTE)


# setup test virtual environment
# ==============================

# command to create a virtual environment
add_custom_command(
    OUTPUT ${venv_dir}
    COMMAND ${Python3_EXECUTABLE} -m venv ${venv_dir}
    COMMENT "Creating virtual environment for test dependencies"
)

# command to upgrade pip in the virtual environment
add_custom_command(
    OUTPUT ${venv_dir}/bin/pip-upgraded
    COMMAND ${venv_dir}/bin/python -m pip install --upgrade pip
    COMMAND ${CMAKE_COMMAND} -E touch ${venv_dir}/bin/pip-upgraded
    DEPENDS ${venv_dir}
    COMMENT "Upgrading pip in virtual environment"
)

# command to install test dependencies into the virtual environment
add_custom_command(
    OUTPUT ${venv_dir}/bin/pytest
    COMMAND ${venv_dir}/bin/pip install -r ${reqs}
    DEPENDS ${venv_dir}/bin/pip-upgraded
    COMMENT "Installing test dependencies"
)

# add a custom target to trigger all the above commands
add_custom_target(
    setup_test_env ALL
    DEPENDS ${venv_dir}/bin/pytest
    COMMENT "Test environment setup complete"
)


# setup test target
# =================

add_custom_target(pyghex_tests)
add_dependencies(pyghex pyghex_tests)
add_dependencies(setup_test_env pyghex_tests)

copy_files(TARGET pyghex_tests DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/fixtures FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/context.py)
copy_files(TARGET pyghex_tests DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/conftest.py)


# register unit tests
# ===================

function(ghex_reg_pytest t)
    copy_files(TARGET pyghex_tests DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/test_${t}.py)
    add_test(
        NAME py_${t}
        COMMAND ${venv_dir}/bin/python -m pytest -s ${CMAKE_CURRENT_BINARY_DIR}/test_${t}.py
        WORKING_DIRECTORY ${pyghex_test_workdir})
    set_tests_properties(py_${t} PROPERTIES RUN_SERIAL ON)
endfunction()

function(ghex_reg_parallel_pytest t n)
    copy_files(TARGET pyghex_tests DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/test_${t}.py)
    add_test(
        NAME py_${t}_parallel
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${MPIEXEC_PREFLAGS}
        ${venv_dir}/bin/python -m pytest -s --with-mpi ${CMAKE_CURRENT_BINARY_DIR}/test_${t}.py
        WORKING_DIRECTORY ${pyghex_test_workdir})
    set_tests_properties(py_${t}_parallel PROPERTIES RUN_SERIAL ON)
endfunction()

ghex_reg_pytest(context)
ghex_reg_parallel_pytest(context 4)

ghex_reg_parallel_pytest(structured_domain_descriptor 4)
ghex_reg_parallel_pytest(structured_pattern 4)

#ghex_reg_pytest(unstructured_domain_descriptor)
ghex_reg_parallel_pytest(unstructured_domain_descriptor 4)
