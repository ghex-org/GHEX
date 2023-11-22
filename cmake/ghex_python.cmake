include(GNUInstallDirs)

set(PYBIND11_CPP_STANDARD -std=c++17)

if (GHEX_BUILD_PYTHON_BINDINGS)

    if (GHEX_WITH_TESTING)
        # create a virtual environment
        # according to https://discourse.cmake.org/t/possible-to-create-a-python-virtual-env-from-cmake-and-then-find-it-with-findpython3/1132
        find_package (Python3 REQUIRED COMPONENTS Interpreter)
        set(venv "${CMAKE_CURRENT_BINARY_DIR}/pyghex_venv")
        set(venv_bin_dir "${venv}/bin")
        set(reqs "${PROJECT_SOURCE_DIR}/bindings/python/requirements-test.txt")
        message("Creating VENV from ${Python3_EXECUTABLE} to ${VENV}")
        execute_process(COMMAND_ECHO STDOUT COMMAND ${Python3_EXECUTABLE} -m venv --system-site-packages ${venv} )
        execute_process(COMMAND_ECHO STDOUT COMMAND ${venv_bin_dir}/pip install -U pip setuptools wheel )
        execute_process(COMMAND_ECHO STDOUT COMMAND ${venv_bin_dir}/pip install -r ${reqs} )
        execute_process(
            COMMAND_ECHO STDOUT
            COMMAND ${Python3_EXECUTABLE} -c "import sys;print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))"
            OUTPUT_VARIABLE python_version
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(venv_site_packages_dir "${venv}/lib/python${python_version}/site-packages")
        #execute_process (COMMAND "${Python3_EXECUTABLE}" -m venv ${venv})
        # Here is the trick
        ## update the environment with VIRTUAL_ENV variable (mimic the activate script)
        set (ENV{VIRTUAL_ENV} "${CMAKE_CURRENT_BINARY_DIR}/pyghex_venv")
        ## change the context of the search
        set (Python3_FIND_VIRTUALENV FIRST)
        ## unset Python3_EXECUTABLE because it is also an input variable (see documentation, Artifacts Specification section)
        unset (Python3_EXECUTABLE)
        ## Launch a new search
        find_package (Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    else()
        #if(DEFINED PYTHON_EXECUTABLE)
        #    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
        #endif()
        #set(Python_FIND_STRATEGY LOCATION)
        find_package (Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    endif()
    message(STATUS ${Python_SITELIB})

    if(${Python3_FOUND})
        set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
        message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
    endif()

    include(ghex_find_python_module)

    find_package(pybind11 REQUIRED PATHS ${Python_SITELIB})

    #set(GHEX_PYTHON_BINDINGS_SOURCE_FILES bindings/python/cpp/src/ghex.cpp
    #    bindings/python/cpp/src/communication_object.cpp
    #    bindings/python/cpp/src/common/coordinate.cpp
    #    bindings/python/cpp/src/structured/regular/domain_descriptor.cpp
    #    bindings/python/cpp/src/structured/regular/field_descriptor.cpp
    #    bindings/python/cpp/src/structured/regular/halo_generator.cpp
    #    bindings/python/cpp/src/transport_layer/communicator.cpp
    #    bindings/python/cpp/src/transport_layer/context.cpp
    #    bindings/python/cpp/src/pattern.cpp
    #    bindings/python/cpp/src/buffer_info.cpp)
    #pybind11_add_module(ghex_py_bindings ${GHEX_PYTHON_BINDINGS_SOURCE_FILES})
    #if(USE_GPU)
    #    set_source_files_properties(${GHEX_PYTHON_BINDINGS_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    #endif()
    #target_include_directories(ghex_py_bindings PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/cpp/include")
    #if (HAVE_MPI4PY)
    #    target_include_directories(ghex_py_bindings PRIVATE "${PY_MPI4PY}/include")
    #    target_compile_definitions(ghex_py_bindings PRIVATE GHEX_ENABLE_MPI4PY)
    #else()
    #    message(WARNING "MPI4PY not found. Python bindings will be compiled without MPI4PY support.\
    #     Make sure mpi4py is importable in the environment in which cmake is invoked to avoid this.")
    #endif()


    #pybind11_add_module(phex ${GHEX_PYTHON_BINDINGS_SOURCE_FILES})

    #if (HAVE_MPI4PY)
    #    target_include_directories(pyghex PRIVATE "${PY_MPI4PY}/include")
    #    target_compile_definitions(pyghex PRIVATE GHEX_ENABLE_MPI4PY)
    #else()
    #endif()

    #target_link_libraries(pyghex PRIVATE ghex)
    ##target_link_libraries(pyghex INTERFACE GridTools::gridtools)
endif()
