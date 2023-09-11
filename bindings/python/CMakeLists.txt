include(ghex_copy_files)

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
    unstructured/domain_descriptor.cpp
    unstructured/halo_generator.cpp
    unstructured/field_descriptor.cpp
    unstructured/pattern.cpp
    unstructured/communication_object.cpp
)
set_source_files_properties(config.cpp PROPERTIES 
    COMPILE_DEFINITIONS GHEX_TRANSPORT_BACKEND=${GHEX_TRANSPORT_BACKEND})

# The Python object
# =================
add_library(pyghex_obj OBJECT ${pyghex_source})
set_property(TARGET pyghex_obj PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(pyghex_obj PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
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


# The Python module
# ==================
# Create the Python module in the build directory.
# The module contains the dynamic library, __init__.py and VERSION information.
set(python_mod_path "${CMAKE_CURRENT_BINARY_DIR}/ghex")
add_library(pyghex MODULE $<TARGET_OBJECTS:pyghex_obj>)
# With this, the full name of the library will be something like:
# _pyghex.cpython-36m-x86_64-linux-gnu.so
set_target_properties(pyghex PROPERTIES
    OUTPUT_NAME _pyghex
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")
set_target_properties(pyghex PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${python_mod_path}")
# This dependency has to be spelt out again, despite being added to
# pyghex_obj because CMake.
target_link_libraries(pyghex PRIVATE pybind11::module)
target_link_libraries(pyghex PRIVATE ghex)
ghex_link_to_oomph(pyghex)
# Copy the required python source files as well
add_custom_target(pyghex_files)
copy_files(TARGET pyghex_files DESTINATION ${python_mod_path} FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${PROJECT_BINARY_DIR}/version.txt
)
copy_files(TARGET pyghex_files DESTINATION ${python_mod_path}/util FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/util/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/util/cpp_wrapper.py
    ${CMAKE_CURRENT_SOURCE_DIR}/util/architecture.py
)
copy_files(TARGET pyghex_files DESTINATION ${python_mod_path}/structured FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/structured/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/structured/index_space.py
    ${CMAKE_CURRENT_SOURCE_DIR}/structured/grid.py
)
copy_files(TARGET pyghex_files DESTINATION ${python_mod_path}/structured/regular FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/structured/regular/__init__.py
)
copy_files(TARGET pyghex_files DESTINATION ${python_mod_path}/unstructured FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/unstructured/__init__.py
)
add_dependencies(pyghex pyghex_files)

# Installation
# ============
# Ask Python where it keeps its system (platform) packages.
file(WRITE "${CMAKE_BINARY_DIR}/install-prefix" "${CMAKE_INSTALL_PREFIX}")
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/where.py"
    INPUT_FILE "${CMAKE_BINARY_DIR}/install-prefix"
    OUTPUT_VARIABLE GHEX_PYTHON_LIB_PATH_DEFAULT_REL
    OUTPUT_STRIP_TRAILING_WHITESPACE)
# convert to absolute path if needed (could be a relative path if ccmake was used)
get_filename_component(GHEX_PYTHON_LIB_PATH_DEFAULT "${GHEX_PYTHON_LIB_PATH_DEFAULT_REL}"
    REALPATH  BASE_DIR "${CMAKE_BINARY_DIR}")
# Default to installing in that path, override with user specified GHEX_PYTHON_LIB_PATH
set(GHEX_PYTHON_LIB_PATH ${GHEX_PYTHON_LIB_PATH_DEFAULT} CACHE PATH "path for installing Python bindings.")
message(STATUS "Python bindings installation path: ${GHEX_PYTHON_LIB_PATH}")

if(DEFINED SKBUILD_PROJECT_NAME)
  # Building wheel through scikit-build-core
  set(_python_module_install_path .)

  # set RPaths
  list(GET CMAKE_PREFIX_PATH 0 scikit_build_core_install_path)
  set_target_properties(pyghex PROPERTIES 
      INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}:${scikit_build_core_install_path}/ghex/${CMAKE_INSTALL_LIBDIR}")
  set_target_properties(ghex PROPERTIES
      INSTALL_RPATH "${scikit_build_core_install_path}/ghex/${CMAKE_INSTALL_LIBDIR}")
  if (GHEX_TRANSPORT_BACKEND STREQUAL "LIBFABRIC")
      set_target_properties(oomph_libfabric PROPERTIES
        INSTALL_RPATH "${scikit_build_core_install_path}/ghex/${CMAKE_INSTALL_LIBDIR}")
  elseif (GHEX_TRANSPORT_BACKEND STREQUAL "UCX")
      set_target_properties(oomph_ucx PROPERTIES
        INSTALL_RPATH "${scikit_build_core_install_path}/ghex/${CMAKE_INSTALL_LIBDIR}")
  else()
      set_target_properties(oomph_mpi PROPERTIES
        INSTALL_RPATH "${scikit_build_core_install_path}/ghex/${CMAKE_INSTALL_LIBDIR}")
  endif()
else()
  set(_python_module_install_path ${GHEX_PYTHON_LIB_PATH}/ghex)
endif()

install(DIRECTORY ${python_mod_path}
    DESTINATION ${_python_module_install_path}/..
    PATTERN "__pycache__" EXCLUDE
)
install(TARGETS pyghex DESTINATION ${_python_module_install_path})
