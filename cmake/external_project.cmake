include(ExternalProject)
include(git_submodule)

# functionality for adding external projects
# Arguments:
# NAME:           project name
# PATH:           project folder
# INTERFACE_NAME: generated cmake target to link against
# LIBS:           library name
# CMAKE_ARGS:     additional cmake arguments
function(add_external_cmake_project)
    # handle named arguments: fills variables EP_NAME, EP_INTERFACE_NAME, EP_LIBS, and EP_CMAKE_ARGS
    set(options OPTIONAL)
    set(oneValueArgs NAME PATH INTERFACE_NAME)
    set(multiValueArgs LIBS CMAKE_ARGS)
    cmake_parse_arguments(EP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(EP_BUILD       "${EP_NAME}-build")
    set(EP_SOURCE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/${EP_PATH}")
    set(EP_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EP_PATH}")

    set(EP_INTERFACE_INCLUDE_DIR "${EP_INSTALL_DIR}/include")
    set(EP_INTERFACE_LIB_NAMES)
    foreach(lib ${EP_LIBS})
        list(APPEND EP_INTERFACE_LIB_NAMES "${EP_INSTALL_DIR}/lib/${lib}")
    endforeach()
    ##set(EP_INTERFACE_LIB_NAME    "${EP_INSTALL_DIR}/lib/${EP_LIB}")
    #list(TRANSFORM ${EP_LIBS} PREPEND "${EP_INSTALL_DIR}/lib/" EP_INTERFACE_LIB_NAMES)
    #foreach(lib ${EP_INTERFACE_LIB_NAMES})
    #    message(STATUS ${lib})
    #endforeach()

    check_git_submodule(${EP_NAME}_sub ${EP_PATH})
    if(${EP_NAME}_sub_avail)
        # populate cmake arguments
        set(EP_ALL_CMAKE_ARGS
            "-DCMAKE_INSTALL_PREFIX=${EP_INSTALL_DIR}"
            "-DCMAKE_INSTALL_LIBDIR=${EP_INSTALL_DIR}/lib"
            "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
        list(APPEND EP_ALL_CMAKE_ARGS ${EP_CMAKE_ARGS})

        # add external project
        ExternalProject_Add(${EP_BUILD}
            # Add dummy DOWNLOAD_COMMAND to stop ExternalProject_Add terminating CMake if the
            # git submodule had not been udpated.
            DOWNLOAD_COMMAND "${CMAKE_COMMAND}" -E echo "Warning: ${EP_SOURCE_DIR} empty or missing."
            #BUILD_BYPRODUCTS "${EP_INTERFACE_LIB_NAME}"
            BUILD_BYPRODUCTS "${EP_INTERFACE_LIB_NAMES}"
            SOURCE_DIR "${EP_SOURCE_DIR}"
            CMAKE_ARGS "${EP_ALL_CMAKE_ARGS}"
            INSTALL_DIR "${EP_INSTALL_DIR}"
        )
        set_target_properties(${EP_BUILD} PROPERTIES EXCLUDE_FROM_ALL TRUE)

        # make top level interface library which links to external project
        add_library(${EP_INTERFACE_NAME} INTERFACE)
        add_dependencies(${EP_INTERFACE_NAME} ${EP_BUILD})
        target_include_directories(${EP_INTERFACE_NAME} INTERFACE ${EP_INTERFACE_INCLUDE_DIR})
        #target_link_libraries(${EP_INTERFACE_NAME} INTERFACE ${EP_INTERFACE_LIB_NAME})
        target_link_libraries(${EP_INTERFACE_NAME} INTERFACE ${EP_INTERFACE_LIB_NAMES})
    else()
        add_error_target(${EP_BUILD}
            "Building ${EP_NAME} library"
            "The git submodule for ${EP_NAME} is not available")
    endif()
endfunction()
