set(pyghex_source
    config.cpp
    context_shim.cpp
    module.cpp
    mpi_comm_shim.cpp
    structured/regular/communication_object.cpp
    structured/regular/domain_descriptor.cpp
    structured/regular/field_descriptor.cpp
    structured/regular/halo_generator.cpp
    structured/regular/pattern.cpp
    unstructured/communication_object.cpp
    unstructured/domain_descriptor.cpp
    unstructured/field_descriptor.cpp
    unstructured/halo_generator.cpp
    unstructured/pattern.cpp
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
link_device_runtime(pyghex_obj)
target_link_libraries(pyghex_obj PUBLIC ghex_common)

# Add GPU support
compile_as_cuda(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} SOURCES ${pyghex_source})

# The Python module
# ==================
# Create the Python module in the build directory.
# The module contains the dynamic library, __init__.py and VERSION information.
add_library(pyghex MODULE $<TARGET_OBJECTS:pyghex_obj>)
# With this, the full name of the library will be something like:
# _pyghex.cpython-36m-x86_64-linux-gnu.so
set_target_properties(pyghex PROPERTIES
    OUTPUT_NAME _pyghex
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    # Choose this particular output directory for testing purposes
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../../ghex")
# This dependency has to be spelt out again, despite being added to
# pyghex_obj because CMake.
target_link_libraries(pyghex PRIVATE pybind11::module)
target_link_libraries(pyghex PRIVATE ghex)
ghex_link_to_oomph(ghex)

# Set RPaths such that the python module is able to find libghex
include(ghex_rpath)
if (SKBUILD_PROJECT_NAME)
    set_target_properties(pyghex PROPERTIES INSTALL_RPATH "${rpath_origin_install_libdir}")
    install(TARGETS pyghex DESTINATION .)
else()
    set_target_properties(pyghex PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    install(TARGETS pyghex DESTINATION ${GHEX_PYTHON_LIB_PATH}/ghex)
endif()
