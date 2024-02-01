
find_package(MPI REQUIRED COMPONENTS CXX)

add_library(gtest_main_mpi ./gtest_main_mpi.cpp)
target_link_libraries(gtest_main_mpi PRIVATE ext-gtest-ghex)
target_link_libraries(gtest_main_mpi PRIVATE MPI::MPI_CXX)

add_library(gtest_main_mpi_mt ./gtest_main_mpi.cpp)
target_compile_definitions(gtest_main_mpi_mt PRIVATE GHEX_TEST_MULTI_THREADED)
target_link_libraries(gtest_main_mpi_mt PRIVATE ext-gtest-ghex)
target_link_libraries(gtest_main_mpi_mt PRIVATE MPI::MPI_CXX)
