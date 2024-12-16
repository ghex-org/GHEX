# rpath_origin for libraries that have their dependencies in the same directory (installed in ${CMAKE_INSTALL_LIBDIR})
# rpath_origin_install_libdir for the pyghex library that has its dependencies in ${CMAKE_INSTALL_LIBDIR} relative to its directory
if(APPLE)
    set(rpath_origin "@loader_path")
    set(rpath_origin_install_libdir "@loader_path/${CMAKE_INSTALL_LIBDIR}")
else()
    set(rpath_origin "$ORIGIN")
    set(rpath_origin_install_libdir "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
endif()
