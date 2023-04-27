# Set up google test as an external project
if (NOT ${GHEX_USE_BUNDLED_OOMPH})
add_external_cmake_project(
    NAME googletest
    PATH ../ext/googletest
    INTERFACE_NAME ext-gtest
    LIBS libgtest.a libgtest_main.a
    CMAKE_ARGS
        "-DCMAKE_BUILD_TYPE=release"
        "-DBUILD_SHARED_LIBS=OFF"
        "-DBUILD_GMOCK=OFF")

# on some systems we need link explicitly against threads
if (TARGET ext-gtest)
    find_package (Threads)
    target_link_libraries(ext-gtest INTERFACE Threads::Threads)
endif()
endif()
