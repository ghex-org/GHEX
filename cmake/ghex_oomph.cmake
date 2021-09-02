if(NOT _oomph_already_fetched)
    find_package(oomph QUIET)
endif()
if(NOT oomph_FOUND)
    set(_oomph_repository "https://github.com/boeschf/oomph.git")
    set(_oomph_tag        "main")
    if(NOT _oomph_already_fetched)
        message(STATUS "Fetching oomph ${_oomph_tag} from ${_oomph_repository}")
    endif()
    include(FetchContent)
    FetchContent_Declare(
        oomph
        GIT_REPOSITORY ${_oomph_repository}
        GIT_TAG        ${_oomph_tag}
    )
    FetchContent_MakeAvailable(oomph)
    set(_oomph_already_fetched ON CACHE INTERNAL "")
endif()
