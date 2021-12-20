include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    #GIT_TAG        release-1.10.0
    GIT_TAG        main
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    # https://github.com/google/googletest/issues/2429
    add_library(GTest::gtest ALIAS gtest)
endif()

