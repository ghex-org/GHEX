if (GHEX_BUILD_FORTRAN)
    enable_language(Fortran)
    set(GHEX_FORTRAN_FP "float" CACHE STRING "Floating-point type")
    set(GHEX_FORTRAN_OPENMP "ON" CACHE BOOL "Compile Fortran bindings with OpenMP")
    set_property(CACHE GHEX_FORTRAN_FP PROPERTY STRINGS "float" "double")
    if(${GHEX_FORTRAN_FP} STREQUAL "float")
        set(GHEX_FORTRAN_FP_KIND 4)
    else()
        set(GHEX_FORTRAN_FP_KIND 8)
    endif()

    find_package(MPI REQUIRED)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ghex_defs.f90.in
        ${CMAKE_CURRENT_BINARY_DIR}/bindings/fhex/ghex_defs.f90 @ONLY)
    install(FILES ${PROJECT_BINARY_DIR}/bindings/fhex/ghex_defs.f90
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bindings/fhex)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ghex_defs.hpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/bindings/fhex/ghex_defs.hpp @ONLY)
    install(FILES ${PROJECT_BINARY_DIR}/bindings/fhex/ghex_defs.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bindings/fhex)
endif()
