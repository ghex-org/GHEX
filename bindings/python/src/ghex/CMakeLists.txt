if(GHEX_WITH_TESTING)
    add_custom_target(pyghex_files)
    add_custom_command(TARGET pyghex_files PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/../../ghex")
    add_custom_command(TARGET pyghex_files PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/version.txt "${CMAKE_CURRENT_BINARY_DIR}/../../ghex")
    add_dependencies(pyghex pyghex_files)
endif()

