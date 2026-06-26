if (SKBUILD_PROJECT_NAME)
    # CMake driven by scikit-build-core
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION . FILES_MATCHING PATTERN "*.py")
    install(FILES ${CMAKE_BINARY_DIR}/version.txt DESTINATION .)
else()
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${GHEX_PYTHON_LIB_PATH} FILES_MATCHING PATTERN "*.py")
    install(FILES ${CMAKE_BINARY_DIR}/version.txt DESTINATION ${GHEX_PYTHON_LIB_PATH}/ghex)
endif()

if(GHEX_WITH_TESTING)
    file(GLOB_RECURSE pyghex_python_files CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

    add_custom_target(pyghex_files ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/../../ghex"
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/version.txt "${CMAKE_CURRENT_BINARY_DIR}/../../ghex"
        DEPENDS ${pyghex_python_files} ${CMAKE_BINARY_DIR}/version.txt
        COMMENT "Refreshing Python binding files for tests")

    add_dependencies(pyghex_files pyghex)
endif()
