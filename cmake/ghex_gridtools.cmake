set(_required_gridtools_version "2.0.0")
if(NOT _gridtools_already_fetched)
    find_package(GridTools ${_required_gridtools_version} QUIET)
endif()
if(NOT GridTools_FOUND)
    set(_gridtools_repository "https://github.com/GridTools/gridtools.git")
    set(_gridtools_tag        "v${_required_gridtools_version}")
    if(NOT _gridtools_already_fetched)
        message(STATUS "Fetching GridTools tag ${_gridtools_tag} from ${_gridtools_repository}")
    endif()
    include(FetchContent)
    FetchContent_Declare(
        gridtools
        GIT_REPOSITORY ${_gridtools_repository}
        GIT_TAG        ${_gridtools_tag}
    )
    FetchContent_MakeAvailable(gridtools)
    set(_gridtools_already_fetched ON CACHE INTERNAL "")
endif()
