cmake_minimum_required(VERSION 3.12.4)

cmake_policy(SET CMP0048 NEW)

project(GHEX VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(MPI REQUIRED)

find_package(GridTools 0.21 REQUIRED)

message(">>>>>>>>>>>>>>>>${GridTools_MODULE_PATH}")

list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

#add_executable( mbianco_gcl2 tests/gcl2_app.cpp )
#target_compile_options(mbianco_gcl2 PRIVATE -DMT_MPI)
#target_link_libraries(mbianco_gcl2 ${exe_LIBS} GridTools::gridtools GridTools::gcl)
#
#add_executable( mbianco_gcl2_os tests/gcl2_app.cpp )
#target_compile_options(mbianco_gcl2_os PRIVATE -DOS_MPI)
#target_link_libraries(mbianco_gcl2_os ${exe_LIBS} ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)
#
#add_executable( mbianco_gcl2_managed tests/gcl2_app.cpp )
#target_link_libraries(mbianco_gcl2_managed ${exe_LIBS} GridTools::gridtools GridTools::gcl)
#
#add_executable( test_iteration_spaces tests/test_iteration_spaces.cpp )
#target_link_libraries(test_iteration_spaces ${exe_LIBS} GridTools::gridtools GridTools::gcl)


add_executable( regular_grid_descriptor tests/regular_grid_descriptor.cpp )
target_link_libraries( regular_grid_descriptor ${exe_LIBS} GridTools::gridtools )

#add_executable( range_loops tests/range_loops.cpp )
#target_link_libraries( range_loops ${exe_LIBS} GridTools::gridtools )
#
#
#add_executable( test_is_multi_arg tests/test_is_multi_arg.cpp )
#target_link_libraries(test_is_multi_arg ${exe_LIBS} GridTools::gridtools GridTools::gcl)
#
#
#add_executable( generic_interfaces tests/generic_interfaces.cpp )
#target_link_libraries(generic_interfaces ${exe_LIBS}  ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)
#
#add_executable( generic_interfaces8 tests/generic_interfaces8.cpp )
#target_link_libraries(generic_interfaces8 ${exe_LIBS}  ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)
#
#add_executable( generic_interfaces1D tests/generic_interfaces1D.cpp )
#target_link_libraries(generic_interfaces1D ${exe_LIBS}  ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)
#
#add_executable( generic_interfaces1D_mt tests/generic_interfaces1D_mt.cpp )
#target_link_libraries(generic_interfaces1D_mt ${exe_LIBS}  ${MPI_CXX_LIBRARIES} GridTools::gridtools GridTools::gcl)
