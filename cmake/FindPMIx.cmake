find_package(PkgConfig QUIET)
pkg_check_modules(PC_PMIX QUIET pmix)

find_path(PMIX_INCLUDE_DIR pmix.h
    HINTS
        ${PMIX_ROOT}  ENV PMIX_ROOT
        ${PMIX_DIR}   ENV PMIX_DIR
    PATH_SUFFIXES include)

find_library(PMIX_LIBRARY NAMES pmix
    HINTS
    ${PMIX_ROOT} ENV PMIX_ROOT
    PATH_SUFFIXES lib lib64)

set(PMIX_LIBRARIES    ${PMIX_LIBRARY} CACHE INTERNAL "")
set(PMIX_INCLUDE_DIRS ${PMIX_INCLUDE_DIR} CACHE INTERNAL "")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PMIx DEFAULT_MSG PMIX_LIBRARY PMIX_INCLUDE_DIR)

#foreach(v LIBFABRIC_ROOT)
#  get_property(_type CACHE ${v} PROPERTY TYPE)
#  if(_type)
#    set_property(CACHE ${v} PROPERTY ADVANCED 1)
#    if("x${_type}" STREQUAL "xUNINITIALIZED")
#      set_property(CACHE ${v} PROPERTY TYPE PATH)
#    endif()
#  endif()
#endforeach()

mark_as_advanced(PMIX_ROOT PMIX_LIBRARY PMIX_INCLUDE_DIR)

