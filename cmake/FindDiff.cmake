

find_program(DIFF_BIN
    NAMES diff
    DOC "path to diff executable"
)
if(NOT DIFF_BIN)
    set(DIFF_FOUND 0)
    message(STATUS "diff not found")
else()
    set(DIFF_FOUND 1)
    message(STATUS "diff found at ${DIFF_BIN}")
    mark_as_advanced(DIFF_BIN)
endif()

