include(GNUInstallDirs)

if (GHEX_BUILD_PYTHON_BINDINGS)
    include(ghex_find_python_module)

    find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)
    set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")

    # Look for the `nanobind` Python module.
    find_python_module(nanobind)

    if (SKBUILD_PROJECT_NAME)
      message(STATUS "Building in pip mode.")
      # Build as a Python package, `nanobind` is a build dependency, so it should be found.
      if(NOT HAVE_NANOBIND)
        message(FATAL_ERROR "Expected that the `nanobind` Python pakage was installed as dependency")
      endif()
      find_package(nanobind CONFIG REQUIRED HINTS "${PY_NANOBIND}/cmake")
    elseif (HAVE_NANOBIND)
      message(STATUS "Building in normal mode but use installed nanobind package.")
      # Normal build and the `nanobind` Python package was found, use it.
      find_package(nanobind CONFIG REQUIRED HINTS "${PY_NANOBIND}/cmake")
    else()
      message(STATUS "Building in normal mode but use system nanobind.")
      # Normal build but no `nanobind` Python package was found, try to localize the one on the system.
      # NOTE: The `CONFIG` is retained for compatibility with the old version, but maybe remove it.
      find_package(nanobind CONFIG REQUIRED)
    endif()

    # Ask Python where it keeps its system (platform) packages.
    file(WRITE "${CMAKE_BINARY_DIR}/install-prefix" "${CMAKE_INSTALL_PREFIX}")
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/where.py"
        INPUT_FILE "${CMAKE_BINARY_DIR}/install-prefix"
        OUTPUT_VARIABLE GHEX_PYTHON_LIB_PATH_DEFAULT_REL
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Convert to absolute path if needed (could be a relative path if ccmake was used)
    get_filename_component(GHEX_PYTHON_LIB_PATH_DEFAULT "${GHEX_PYTHON_LIB_PATH_DEFAULT_REL}"
        REALPATH  BASE_DIR "${CMAKE_BINARY_DIR}")

    # Default to installing in that path, override with user specified GHEX_PYTHON_LIB_PATH
    set(GHEX_PYTHON_LIB_PATH ${GHEX_PYTHON_LIB_PATH_DEFAULT} CACHE PATH "path for installing Python bindings.")
    message(STATUS "Python bindings installation path: ${GHEX_PYTHON_LIB_PATH}")

endif()
