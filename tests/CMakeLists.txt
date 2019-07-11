set(_tests mpi_allgather communication_object)

foreach (_t ${_tests})
    add_executable(${_t} ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools Boost::mpi)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_t} ${MPIEXEC_POSTFLAGS}
    )
endforeach()


add_executable(communication_object_2_serial communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_include_directories(communication_object_2_serial PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_serial MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_serial
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} communication_object_2_serial ${MPIEXEC_POSTFLAGS}
)

add_executable(communication_object_2_serial_2 communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_compile_definitions(communication_object_2_serial_2 PUBLIC SERIAL_SPLIT) 
target_include_directories(communication_object_2_serial_2 PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_serial_2 MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_serial_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} communication_object_2_serial_2 ${MPIEXEC_POSTFLAGS}
)

add_executable(communication_object_2_threads communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_compile_definitions(communication_object_2_threads PUBLIC MULTI_THREADED_EXCHANGE) 
target_compile_definitions(communication_object_2_threads PUBLIC MULTI_THREADED_EXCHANGE_THREADS) 
target_include_directories(communication_object_2_threads PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_threads MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_threads
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} communication_object_2_threads ${MPIEXEC_POSTFLAGS}
)

add_executable(communication_object_2_async_async communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_compile_definitions(communication_object_2_async_async PUBLIC MULTI_THREADED_EXCHANGE) 
target_compile_definitions(communication_object_2_async_async PUBLIC MULTI_THREADED_EXCHANGE_ASYNC_ASYNC) 
target_include_directories(communication_object_2_async_async PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_async_async MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_async_async
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} communication_object_2_async_async ${MPIEXEC_POSTFLAGS}
)

add_executable(communication_object_2_async_deferred communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_compile_definitions(communication_object_2_async_deferred PUBLIC MULTI_THREADED_EXCHANGE) 
target_compile_definitions(communication_object_2_async_deferred PUBLIC MULTI_THREADED_EXCHANGE_ASYNC_DEFERRED) 
target_include_directories(communication_object_2_async_deferred PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_async_deferred MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_async_deferred
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} communication_object_2_async_deferred ${MPIEXEC_POSTFLAGS}
)

add_executable(communication_object_2_async_async_wait communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_compile_definitions(communication_object_2_async_async_wait PUBLIC MULTI_THREADED_EXCHANGE) 
target_compile_definitions(communication_object_2_async_async_wait PUBLIC MULTI_THREADED_EXCHANGE_ASYNC_ASYNC_WAIT) 
target_include_directories(communication_object_2_async_async_wait PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_async_async_wait MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_async_async_wait
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} communication_object_2_async_async_wait ${MPIEXEC_POSTFLAGS}
)
