cmake_minimum_required(VERSION 3.12.4)

cmake_policy(SET CMP0048 NEW)

project(GHEX VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(MPI REQUIRED)

find_package(GridTools REQUIRED)

add_library(GHEX_libs INTERFACE)
target_include_directories(GHEX_libs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
target_link_libraries(GHEX_libs INTERFACE GridTools::gridtools)

list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#add_executable(mbianco_gcl2_os tests/gcl2_app.cpp)
#target_compile_options(mbianco_gcl2_os PRIVATE -DOS_MPI)
#target_link_libraries(mbianco_gcl2_os ${exe_LIBS} ${MPI_CXX_LIBRARIES} GHEX_libs)
