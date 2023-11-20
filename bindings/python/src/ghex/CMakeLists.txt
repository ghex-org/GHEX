if(GHEX_WITH_TESTING)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${venv_site_packages_dir})
    install(FILES ${CMAKE_BINARY_DIR}/version.txt DESTINATION ${venv_site_packages_dir}/ghex)
endif()