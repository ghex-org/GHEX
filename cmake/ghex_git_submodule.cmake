function(update_git_submodules)
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE OOMPH_GIT_SUBMOD_RESULT)
        if(NOT OOMPH_GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${OOMPH_GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endfunction()
        
# Call to ensure that the git submodule in location `path` is loaded.
# If the submodule is not loaded, an error message that describes
# how to update the submodules is printed.
# Sets the variable name_avail to `ON` if the submodule is available,
# or `OFF` otherwise.

function(check_git_submodule name path)
    set(success_var "${name}_avail")
    set(${success_var} ON PARENT_SCOPE)

    get_filename_component(dotgit "${path}/.git" ABSOLUTE)
    if(NOT EXISTS ${dotgit})
        message(
            "\nThe git submodule for ${name} is not available in ${path}.\n"
            "To check out all submodules use the following commands:\n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or download submodules recursively when checking out:\n"
            "    git clone --recursive https://github.com/ghex-org/ghex.git\n"
        )

        # if the repository was not available, and git failed, set AVAIL to false
        set(${success_var} OFF PARENT_SCOPE)
    endif()
endfunction()
