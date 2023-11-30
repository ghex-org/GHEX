if (SKBUILD_PROJECT_NAME)
    # CMake driven by scikit-build-core
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION .. FILES_MATCHING PATTERN "*.py")
    install(FILES ${CMAKE_BINARY_DIR}/version.txt DESTINATION .)
else()
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${GHEX_PYTHON_LIB_PATH} FILES_MATCHING PATTERN "*.py")
    install(FILES ${CMAKE_BINARY_DIR}/version.txt DESTINATION ${GHEX_PYTHON_LIB_PATH}/ghex)
endif()

if(GHEX_WITH_TESTING)
    add_custom_target(pyghex_files)
    add_custom_command(TARGET pyghex_files PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/../../ghex")
    add_custom_command(TARGET pyghex_files PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/version.txt "${CMAKE_CURRENT_BINARY_DIR}/../../ghex")
    add_dependencies(pyghex pyghex_files)
endif()
