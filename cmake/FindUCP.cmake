find_package(PkgConfig QUIET)
pkg_check_modules(PC_UCX QUIET ucx)

find_path(UCP_INCLUDE_DIR ucp/api/ucp.h
    HINTS
    ${UCX_ROOT}  ENV UCX_ROOT
    ${UCX_DIR}   ENV UCX_DIR
    PATH_SUFFIXES include)

find_library(UCP_LIBRARY NAMES ucp
    HINTS
    ${UCX_ROOT} ENV UCX_ROOT
    PATH_SUFFIXES lib lib64)

set(UCP_LIBRARIES    ${UCP_LIBRARY} CACHE INTERNAL "")
set(UCP_INCLUDE_DIRS ${UCP_INCLUDE_DIR} CACHE INTERNAL "")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UCP DEFAULT_MSG UCP_LIBRARY UCP_INCLUDE_DIR)

mark_as_advanced(UCX_ROOT UCP_LIBRARY UCP_INCLUDE_DIR)
