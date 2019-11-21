

add_executable( threads threads.cpp )
target_include_directories( threads PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries( threads  GridTools::gridtools MPI::MPI_CXX gtest_main_mt)
add_test(
    NAME threads.cpp
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} threads ${MPIEXEC_POSTFLAGS}
)


add_executable( threads_omp threads.cpp )
target_include_directories( threads_omp PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_compile_options( threads_omp PRIVATE -fopenmp )
target_link_libraries( threads_omp  GridTools::gridtools MPI::MPI_CXX gtest_main_mt)
add_test(
    NAME threads_omp.cpp
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} threads_omp ${MPIEXEC_POSTFLAGS}
)
