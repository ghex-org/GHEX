include(GNUInstallDirs)

if (GHEX_BUILD_PYTHON_BINDINGS)

    find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)

    set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")

    include(ghex_find_python_module)
    find_python_module(nanobind REQUIRED)
    find_package(nanobind CONFIG REQUIRED HINTS "${PY_NANOBIND}/cmake")
    message(STATUS "`python-nanobind` version ${nanobind_VERSION} found.")

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

endif()
