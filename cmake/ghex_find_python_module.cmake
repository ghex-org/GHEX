# https://cmake.org/pipermail/cmake/2011-January/041666.html
include(FindPackageHandleStandardArgs)
function(find_python_module module)
    string(TOUPPER ${module} module_upper)

    if(NOT PY_${module_upper})
        if(ARGC GREATER 1 AND ARGV1 STREQUAL "REQUIRED")
            set(${module}_FIND_REQUIRED TRUE)
        endif()

        # A module's location is usually a directory, but for binary modules
        # it's a .so file.
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
            "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
            RESULT_VARIABLE _${module}_status
            OUTPUT_VARIABLE _${module}_location
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
            "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__version__))"
            OUTPUT_VARIABLE _${module}_version
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        message(STATUS "python-exe = ${PYTHON_EXECUTABLE}")
        message(STATUS "status = ${_${module}_status}")
        message(STATUS "location = ${_${module}_location}")
        message(STATUS "version = ${_${module}_version}")

        if(NOT _${module}_status)
            set(HAVE_${module_upper} ON CACHE INTERNAL "Python module available")
            set(PY_${module_upper} ${_${module}_location} CACHE STRING "Location of Python module ${module}")
            set(PY_${module_upper}_VERSION ${_${module}_version} CACHE STRING "Version of Python module ${module}")
        else()
            set(HAVE_${module_upper} OFF CACHE INTERNAL "Python module available")
        endif()
    endif(NOT PY_${module_upper})

    find_package_handle_standard_args(PY_${module} DEFAULT_MSG PY_${module_upper})
endfunction(find_python_module)
