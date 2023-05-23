
set(pyghex_source
    module.cpp
    config.cpp
    mpi_comm_shim.cpp
    context_shim.cpp
    structured/regular/domain_descriptor.cpp
    structured/regular/halo_generator.cpp
    structured/regular/field_descriptor.cpp
    structured/regular/pattern.cpp
    structured/regular/communication_object.cpp
)
set_source_files_properties(config.cpp PROPERTIES 
    COMPILE_DEFINITIONS GHEX_TRANSPORT_BACKEND=${GHEX_TRANSPORT_BACKEND})

# The Python object
# =================
add_library(pyghex_obj OBJECT ${pyghex_source})
target_include_directories(pyghex_obj PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
#set_target_properties(pyghex_obj PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(pyghex_obj PRIVATE pybind11::module)
ghex_target_compile_options(pyghex_obj)
link_device_runtime(pyghex_obj})
target_link_libraries(pyghex_obj PUBLIC ghex_common)

# Add support for mpi4py if available.
find_python_module(mpi4py)
if (HAVE_MPI4PY)
    target_include_directories(pyghex_obj PRIVATE "${PY_MPI4PY}/include")
    target_compile_definitions(pyghex_obj PRIVATE -DGHEX_ENABLE_MPI4PY)
endif()


# The Python library
# ==================
add_library(pyghex MODULE $<TARGET_OBJECTS:pyghex_obj>)
# With this, the full name of the library will be something like:
#   _pyghex.cpython-36m-x86_64-linux-gnu.so
set_target_properties(pyghex PROPERTIES
    OUTPUT_NAME _pyghex
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")
# This dependency has to be spelt out again, despite being added to
# pyghex_obj because CMake.
target_link_libraries(pyghex PRIVATE pybind11::module)
target_link_libraries(pyghex PRIVATE ghex)
ghex_link_to_oomph(pyghex)


# A helper function to copy files and make them part of a custom target so that they are copied
# prior to "building"
# Arguments:
# TARGET
# DESTINATION
# FILES
function(copy_files)
    set(options OPTIONAL)
    set(oneValueArgs TARGET DESTINATION)
    set(multiValueArgs FILES)
    cmake_parse_arguments(CF "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_custom_command(TARGET ${CF_TARGET} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CF_DESTINATION})
    foreach(file ${CF_FILES})
        add_custom_command(TARGET ${CF_TARGET} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${file} ${CF_DESTINATION})
    endforeach()
endfunction()


# The Python module
# ==================
# Create the Python module in the build directory.
# The module contains the dynamic library, __init__.py and VERSION information.
set(ghex_python_path "${CMAKE_CURRENT_BINARY_DIR}")
set(python_mod_path "${CMAKE_CURRENT_BINARY_DIR}/ghex")
set_target_properties(pyghex PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${python_mod_path}")

add_custom_target(pyghex_module)
copy_files(TARGET pyghex_module DESTINATION ${python_mod_path} FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${PROJECT_BINARY_DIR}/version.txt
)
copy_files(TARGET pyghex_module DESTINATION ${python_mod_path}/util FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/util/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/util/cpp_wrapper.py
    ${CMAKE_CURRENT_SOURCE_DIR}/util/architecture.py
)
copy_files(TARGET pyghex_module DESTINATION ${python_mod_path}/structured FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/structured/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/structured/index_space.py
    ${CMAKE_CURRENT_SOURCE_DIR}/structured/grid.py
)
copy_files(TARGET pyghex_module DESTINATION ${python_mod_path}/structured/regular FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/structured/regular/__init__.py
)
add_dependencies(pyghex pyghex_module)


# The Python tests
# ================
if (GHEX_WITH_TESTING)
    add_custom_target(pyghex_tests)
    add_dependencies(pyghex_module pyghex_tests)
    copy_files(TARGET pyghex_tests DESTINATION ${python_mod_path}/../tests/fixtures FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/context.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/pyghex_path.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/tests/fixtures/pyghex_path.py @ONLY)

    function(ghex_reg_pytest t)
        copy_files(TARGET pyghex_tests DESTINATION ${python_mod_path}/../tests FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${t}.py)
        add_test(
            NAME py_${t}
            COMMAND ${PYTHON_EXECUTABLE} -m pytest -s ${CMAKE_CURRENT_BINARY_DIR}/tests/test_${t}.py)
    endfunction()
    
    function(ghex_reg_parallel_pytest t n)
        copy_files(TARGET pyghex_tests DESTINATION ${python_mod_path}/../tests FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${t}.py)
        add_test(
            NAME py_${t}_parallel
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${MPIEXEC_PREFLAGS}
            ${PYTHON_EXECUTABLE} -m pytest -s --with-mpi ${CMAKE_CURRENT_BINARY_DIR}/tests/test_${t}.py)
    endfunction()

    #ghex_reg_pytest(simple)
    #ghex_reg_parallel_pytest(simple 4)
    ghex_reg_pytest(context)
    ghex_reg_parallel_pytest(context 4)
    ghex_reg_parallel_pytest(domain_descriptor 4)
    ghex_reg_parallel_pytest(pattern 4)
    #ghex_reg_parallel_pytest(pattern 4)

    ##configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tests/__init__.py.in
    ##${CMAKE_CURRENT_BINARY_DIR}/tests/__init__.py @ONLY)
    ##add_subdirectory(tests)
    ##PYTHONPATH="${CMAKE_CURRENT_BINARY_DIR}"
    #add_test(NAME test_simple_parallel
    #    #WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests
    #    #COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/tests/test_simple.py)
    #    #COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_BINARY_DIR})
    #    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${PYTHON_EXECUTABLE} -m pytest --with-mpi ${CMAKE_CURRENT_BINARY_DIR})
    #add_test(NAME test_simple
    #    COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_BINARY_DIR})
endif()
