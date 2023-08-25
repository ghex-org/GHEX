if (GHEX_BUILD_FORTRAN)
    add_subdirectory(fhex)
endif()

if (GHEX_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
