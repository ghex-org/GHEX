include(ghex_compile_options)

add_library(fhex SHARED)

target_include_directories(fhex PRIVATE
    $<INSTALL_INTERFACE:..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>)

ghex_target_compile_options(fhex)
target_link_libraries(fhex PUBLIC ghex_common)
target_link_libraries(fhex PRIVATE ghex)
target_link_libraries(fhex PUBLIC MPI::MPI_Fortran)

target_sources(fhex PRIVATE obj_wrapper.cpp)
target_sources(fhex PRIVATE context_bind.cpp)
target_sources(fhex PRIVATE structured_staged_bind.cpp)

target_sources(fhex PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/ghex_defs.f90)
target_sources(fhex PUBLIC ghex_mod.f90)
target_sources(fhex PUBLIC ghex_structured_mod.f90)

install(TARGETS fhex
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_defs.mod
#        ${CMAKE_CURRENT_BINARY_DIR}/ghex_cubed_sphere_mod.mod
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_mod.mod
        ${CMAKE_CURRENT_BINARY_DIR}/ghex_structured_mod.mod
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/fhex)
