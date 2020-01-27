find_package(PkgConfig QUIET)
pkg_check_modules(PC_PMIX QUIET pmix)

find_path(PMIX_INCLUDE_DIR pmix.h
    HINTS
    ${PMIX_ROOT}  ENV PMIX_ROOT
    ${PMIX_DIR}   ENV PMIX_DIR
    PATH_SUFFIXES include)

find_library(PMIX_LIBRARY HINT ${PMIX_DIR} NAMES pmix
    HINTS
    ${PMIX_ROOT} ENV PMIX_ROOT
    ${PMIX_DIR}  ENV PMIX_DIR
    PATH_SUFFIXES lib lib64)

set(PMIX_LIBRARIES    ${PMIX_LIBRARY} CACHE INTERNAL "")
set(PMIX_INCLUDE_DIRS ${PMIX_INCLUDE_DIR} CACHE INTERNAL "")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PMIx DEFAULT_MSG PMIX_LIBRARY PMIX_INCLUDE_DIR)

mark_as_advanced(PMIX_ROOT PMIX_LIBRARY PMIX_INCLUDE_DIR)

if(NOT TARGET PMIx::libpmix AND PMIx_FOUND)
  add_library(PMIx::libpmix SHARED IMPORTED)
  set_target_properties(PMIx::libpmix PROPERTIES
    IMPORTED_LOCATION ${PMIX_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${PMIX_INCLUDE_DIR}
  )
endif()
