cmake_minimum_required(VERSION 3.12.4)

cmake_policy(SET CMP0048 NEW)

project(GHEX VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(MPI REQUIRED)

find_package(GridTools REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(communication_object tests/communication_object.cpp)
target_link_libraries(communication_object ${exe_LIBS} ${MPI_CXX_LIBRARIES})
