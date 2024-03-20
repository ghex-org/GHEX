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
        execute_process(COMMAND_ECHO STDOUT COMMAND ${Python3_EXECUTABLE} -m venv ${venv} )
        execute_process(COMMAND_ECHO STDOUT COMMAND ${venv_bin_dir}/pip install -U pip setuptools wheel pybind11-stubgen)
        execute_process(COMMAND_ECHO STDOUT COMMAND ${venv_bin_dir}/pip install -r ${reqs} )

        #execute_process (COMMAND "${Python3_EXECUTABLE}" -m venv ${venv})
        # Here is the trick
        ## update the environment with VIRTUAL_ENV variable (mimic the activate script)
        set (ENV{VIRTUAL_ENV} "${venv}")
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

    if(${Python3_FOUND})
        set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
    endif()

    include(ghex_find_python_module)
    find_package(pybind11 REQUIRED PATHS ${Python_SITELIB})

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

    if (GHEX_WITH_TESTING)
        # GHEX_PYTHON_LIB_PATH_DEFAULT has the form /some/abs/install/path/<lib>/<python>/site-packages
        # where <lib> is either lib or lib64, and <python> is python3.x
        cmake_path(GET GHEX_PYTHON_LIB_PATH_DEFAULT PARENT_PATH GHEX_PYTHON_LIB_PATH_DEFAULT-1)
        cmake_path(GET GHEX_PYTHON_LIB_PATH_DEFAULT-1 PARENT_PATH GHEX_PYTHON_LIB_PATH_DEFAULT-2)
        cmake_path(GET GHEX_PYTHON_LIB_PATH_DEFAULT-2 PARENT_PATH GHEX_PYTHON_LIB_PATH_DEFAULT-3)
        cmake_path(RELATIVE_PATH GHEX_PYTHON_LIB_PATH_DEFAULT BASE_DIRECTORY ${GHEX_PYTHON_LIB_PATH_DEFAULT-3} OUTPUT_VARIABLE result)
        cmake_path(ABSOLUTE_PATH result BASE_DIRECTORY ${venv} NORMALIZE OUTPUT_VARIABLE venv_site_packages_dir)
    endif()

endif()
