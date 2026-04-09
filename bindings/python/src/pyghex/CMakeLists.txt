set(CMAKE_CXX_STANDARD 17)

set(pyghex_source
    config.cpp
    context_shim.cpp
    module.cpp
    mpi_comm_shim.cpp
    py_dtype_to_cpp_name.cpp
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
ghex_target_compile_options(pyghex_obj)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(pyghex_obj PRIVATE -fbracket-depth=512)
endif()
link_device_runtime(pyghex_obj)
target_link_libraries(pyghex_obj PUBLIC ghex_common)

# Add GPU support
compile_as_cuda(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} SOURCES ${pyghex_source})

# The Python module
# ==================
# Create the Python module in the build directory.
# The module contains the dynamic library, __init__.py and VERSION information.
nanobind_add_module(pyghex $<TARGET_OBJECTS:pyghex_obj>)

target_link_libraries(pyghex_obj PRIVATE nanobind-static)
# Choose this particular output directory for testing purposes
set_target_properties(pyghex PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../../ghex/")
# This dependency has to be spelt out again, despite being added to
# pyghex_obj because CMake.
target_link_libraries(pyghex PRIVATE ghex)
ghex_link_to_oomph(ghex)

# Generate the type stub if version permits it.
if (${nanobind_VERSION} VERSION_GREATER "2.3")
    nanobind_add_stub(
        pyghex_stub
        MODULE pyghex
        OUTPUT pyghex.pyi
        RECURSIVE  # We need to specify `RECURSIVE` to enable `OUTPUT_PATH`.
        PYTHON_PATH $<TARGET_FILE_DIR:pyghex>
        OUTPUT_PATH $<TARGET_FILE_DIR:pyghex>
	DEPENDS pyghex
    )
endif()

# Set RPaths such that the python module is able to find libghex
include(ghex_rpath)
if (SKBUILD_PROJECT_NAME)
    set_target_properties(pyghex PROPERTIES INSTALL_RPATH "${rpath_origin_install_libdir}")
    install(TARGETS pyghex DESTINATION .)
else()
    set_target_properties(pyghex PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    install(TARGETS pyghex DESTINATION ${GHEX_PYTHON_LIB_PATH}/ghex)
endif()
