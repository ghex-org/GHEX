
# ---------------------------------------------------------------------
# interface library
# ---------------------------------------------------------------------
add_library(ghex_common INTERFACE)
add_library(GHEX::ghex ALIAS ghex_common)

# ---------------------------------------------------------------------
# shared library
# ---------------------------------------------------------------------
add_library(ghex SHARED)
add_library(GHEX::lib ALIAS ghex)
target_link_libraries(ghex PUBLIC ghex_common)
ghex_target_compile_options(ghex)

# ---------------------------------------------------------------------
# install rules
# ---------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ghex_common ghex
    EXPORT ghex-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
