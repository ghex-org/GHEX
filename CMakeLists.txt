cmake_minimum_required(VERSION 3.12.4)

cmake_policy(SET CMP0048 NEW)

project(GHEX VERSION 0.1 LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(MPI REQUIRED)

#find_package(GridTools 0.22 REQUIRED)
find_package(GridTools REQUIRED)

add_library(GHEX_libs INTERFACE)
target_include_directories(GHEX_libs INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>
                          )
target_link_libraries(GHEX_libs INTERFACE GridTools::gridtools  ${MPI_CXX_LIBRARIES})

list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable( mbianco_gcl2 tests/gcl2_app.cpp )
target_compile_options(mbianco_gcl2 PRIVATE -DMT_MPI)
target_link_libraries(mbianco_gcl2 ${exe_LIBS} GHEX_libs)

add_executable( mbianco_gcl2_os tests/gcl2_app.cpp )
target_compile_options(mbianco_gcl2_os PRIVATE -DOS_MPI)
target_link_libraries(mbianco_gcl2_os ${exe_LIBS} GHEX_libs)

# segfaults for me, maybe gridtools not configured properly
#add_executable( mbianco_gcl2_managed tests/gcl2_app.cpp )
#target_link_libraries(mbianco_gcl2_managed ${exe_LIBS} GHEX_libs)

add_executable( test_iteration_spaces tests/test_iteration_spaces.cpp )
target_link_libraries(test_iteration_spaces ${exe_LIBS} GHEX_libs)

add_executable( regular_grid_descriptor tests/regular_grid_descriptor.cpp )
target_link_libraries( regular_grid_descriptor ${exe_LIBS} GHEX_libs )

add_executable( range_loops tests/range_loops.cpp )
target_link_libraries( range_loops ${exe_LIBS} GHEX_libs )

add_executable( test_is_multi_arg tests/test_is_multi_arg.cpp )
target_link_libraries(test_is_multi_arg ${exe_LIBS} GHEX_libs)

add_executable( generic_interfaces tests/generic_interfaces.cpp )
target_link_libraries(generic_interfaces ${exe_LIBS}  GHEX_libs)

add_executable( generic_grid_descriptor tests/generic_grid_descriptor.cpp )
target_link_libraries(generic_grid_descriptor ${exe_LIBS}  GHEX_libs)

add_executable( unstructured_grid tests/unstructured_grid.cpp )
target_link_libraries(unstructured_grid ${exe_LIBS}  GHEX_libs)


add_executable( halo_exchange_3D_generic tests/test_halo_exchange_3D_generic.cpp )
target_link_libraries(halo_exchange_3D_generic ${exe_LIBS}  GHEX_libs)

# this is not working -> iteration space concept not followed
#add_executable( generic_interfaces8 tests/generic_interfaces8.cpp )
#target_link_libraries(generic_interfaces8 ${exe_LIBS}  GHEX_libs)

# this is not working -> iteration space concept not followed
#add_executable( generic_interfaces1D tests/generic_interfaces1D.cpp )
#target_link_libraries(generic_interfaces1D ${exe_LIBS}  GHEX_libs)

# this is not working -> iteration space concept not followed
#add_executable( generic_interfaces1D_mt tests/generic_interfaces1D_mt.cpp )
#target_link_libraries(generic_interfaces1D_mt ${exe_LIBS}  GHEX_libs)
