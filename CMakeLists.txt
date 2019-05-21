cmake_minimum_required(VERSION 3.12.4)

cmake_policy(SET CMP0048 NEW)

project(GHEX VERSION 0.1 LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(MPI REQUIRED)

#find_package(GridTools 0.22 REQUIRED)
find_package(GridTools REQUIRED)

#find_package(Boost REQUIRED)
find_package(Boost COMPONENTS mpi REQUIRED)

add_library(GHEX_libs INTERFACE)
target_include_directories(GHEX_libs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
target_link_libraries(GHEX_libs INTERFACE GridTools::gridtools)

message(">>>>>>>>>>>>>>>>${GridTools_MODULE_PATH}")

list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable( mpi_all_to_all tests/mpi_all_to_all.cpp )
target_compile_options(mpi_all_to_all PRIVATE -DOS_MPI)
target_link_libraries(mpi_all_to_all ${exe_LIBS} GridTools::gridtools GridTools::gcl Boost::mpi )

