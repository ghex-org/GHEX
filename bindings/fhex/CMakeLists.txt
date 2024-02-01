
add_library(fhex SHARED)
set_target_properties(fhex PROPERTIES LINKER_LANGUAGE Fortran)
set_target_properties(fhex PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Mod)

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
target_sources(fhex PRIVATE unstructured_bind.cpp)

target_sources(fhex PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/ghex_defs.f90)
target_sources(fhex PUBLIC ghex_mod.f90)
target_sources(fhex PUBLIC ghex_structured_mod.f90)
target_sources(fhex PUBLIC ghex_unstructured_mod.f90)

install(TARGETS fhex
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/Mod/GHEX_DEFS.mod
            ${CMAKE_CURRENT_BINARY_DIR}/Mod/GHEX_MOD.mod
            ${CMAKE_CURRENT_BINARY_DIR}/Mod/GHEX_STRUCTURED_MOD.mod
            ${CMAKE_CURRENT_BINARY_DIR}/Mod/GHEX_UNSTRUCTURED_MOD.mod
        DESTINATION
            ${CMAKE_INSTALL_INCLUDEDIR}/bindings/fhex)
else()
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/Mod/ghex_defs.mod
            ${CMAKE_CURRENT_BINARY_DIR}/Mod/ghex_mod.mod
            ${CMAKE_CURRENT_BINARY_DIR}/Mod/ghex_structured_mod.mod
            ${CMAKE_CURRENT_BINARY_DIR}/Mod/ghex_unstructured_mod.mod
        DESTINATION
            ${CMAKE_INSTALL_INCLUDEDIR}/bindings/fhex)
endif()
