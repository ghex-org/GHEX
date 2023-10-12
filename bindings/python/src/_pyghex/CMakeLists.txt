include(ghex_copy_files)
include(ghex_device)

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
link_device_runtime(pyghex_obj)
target_link_libraries(pyghex_obj PUBLIC ghex_common)

# Add GPU support
if (GHEX_USE_GPU)
    if (ghex_gpu_mode STREQUAL "cuda")
        set_source_files_properties(${pyghex_source} PROPERTIES LANGUAGE CUDA)
        if (${CMAKE_VERSION} VERSION_LESS "3.24.0" OR GHEX_GPU_ARCH)
            set_target_properties(pyghex_obj PROPERTIES CUDA_ARCHITECTURES ${GHEX_GPU_ARCH})
        else()
            set_target_properties(pyghex_obj PROPERTIES CUDA_ARCHITECTURES native)
        endif()
    elseif (ghex_gpu_mode STREQUAL "hip")
        find_package(hip REQUIRED)
        target_link_libraries(pyghex_obj PRIVATE hip::device)
        # workaround for clang-hip
        #target_compile_options(pyghex_obj PRIVATE $<BUILD_INTERFACE:-std=c++14>)
    endif()
endif()

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
add_library(pyghex MODULE $<TARGET_OBJECTS:pyghex_obj>)
# With this, the full name of the library will be something like:
# _pyghex.cpython-36m-x86_64-linux-gnu.so
set_target_properties(pyghex PROPERTIES
    OUTPUT_NAME _pyghex
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")
# This dependency has to be spelt out again, despite being added to
# pyghex_obj because CMake.
target_link_libraries(pyghex PRIVATE pybind11::module)
target_link_libraries(pyghex PRIVATE ghex)
ghex_link_to_oomph(pyghex)

# Set RPATH
execute_process(
    COMMAND "python -c 'import site; print(site.getsitepackages()[0])'"
    OUTPUT_VARIABLE ghex_python_lib_installation_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE)
set_target_properties(pyghex PROPERTIES INSTALL_RPATH ${ghex_python_lib_installation_dir})

install(TARGETS pyghex DESTINATION ${ghex_python_lib_installation_dir})