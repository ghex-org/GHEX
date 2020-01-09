

find_program(SED_BIN
    NAMES sed
    DOC "path to diff executable"
)
if(NOT SED_BIN)
    set(SED_FOUND 0)
    message(STATUS "sed not found")
else()
    set(SED_FOUND 1)
    message(STATUS "sed found at ${SED_BIN}")
    mark_as_advanced(SED_BIN)
endif()
