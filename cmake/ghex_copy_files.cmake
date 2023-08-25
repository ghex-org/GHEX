# A helper function to copy files and make them part of a custom target so that they are copied
# prior to "building"
# Arguments:
# TARGET
# DESTINATION
# FILES
function(copy_files)
    set(options OPTIONAL)
    set(oneValueArgs TARGET DESTINATION)
    set(multiValueArgs FILES)
    cmake_parse_arguments(CF "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_custom_command(TARGET ${CF_TARGET} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CF_DESTINATION})
    foreach(file ${CF_FILES})
        add_custom_command(TARGET ${CF_TARGET} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${file} ${CF_DESTINATION})
    endforeach()
endfunction()
