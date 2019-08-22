
if(USE_MPI_WITH_UCX_IN_TESTS)
    set(_ucx_params -mca pml ucx)
else()
    set(_ucx_params )
endif()

set(_tests mpi_allgather communication_object)

foreach (_t ${_tests})
    add_executable(${_t} ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools Boost::mpi)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} ${_t} ${MPIEXEC_POSTFLAGS}
    )
endforeach()


add_executable(communication_object_2_serial communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_include_directories(communication_object_2_serial PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_serial MPI::MPI_CXX GridTools::gridtools Boost::mpi)
target_compile_definitions(communication_object_2_serial PUBLIC GHEX_EMULATE_GPU) 
add_test(
    NAME communication_object_2_serial
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_serial ${MPIEXEC_POSTFLAGS}
)

add_executable(communication_object_2_serial_2 communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_compile_definitions(communication_object_2_serial_2 PUBLIC SERIAL_SPLIT)
target_include_directories(communication_object_2_serial_2 PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_serial_2 MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_serial_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_serial_2 ${MPIEXEC_POSTFLAGS}
)

add_executable(communication_object_2_threads communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_compile_definitions(communication_object_2_threads PUBLIC MULTI_THREADED_EXCHANGE)
target_compile_definitions(communication_object_2_threads PUBLIC MULTI_THREADED_EXCHANGE_THREADS)
target_include_directories(communication_object_2_threads PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_threads MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_threads
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_threads ${MPIEXEC_POSTFLAGS}
)

add_executable(communication_object_2_async_async communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_compile_definitions(communication_object_2_async_async PUBLIC MULTI_THREADED_EXCHANGE)
target_compile_definitions(communication_object_2_async_async PUBLIC MULTI_THREADED_EXCHANGE_ASYNC_ASYNC)
target_include_directories(communication_object_2_async_async PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_async_async MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_async_async
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_async_async ${MPIEXEC_POSTFLAGS}
)

add_executable(communication_object_2_async_deferred communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_compile_definitions(communication_object_2_async_deferred PUBLIC MULTI_THREADED_EXCHANGE)
target_compile_definitions(communication_object_2_async_deferred PUBLIC MULTI_THREADED_EXCHANGE_ASYNC_DEFERRED)
target_include_directories(communication_object_2_async_deferred PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_async_deferred MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_async_deferred
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_async_deferred ${MPIEXEC_POSTFLAGS}
)

add_executable(communication_object_2_async_async_wait communication_object_2.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_compile_definitions(communication_object_2_async_async_wait PUBLIC MULTI_THREADED_EXCHANGE)
target_compile_definitions(communication_object_2_async_async_wait PUBLIC MULTI_THREADED_EXCHANGE_ASYNC_ASYNC_WAIT)
target_include_directories(communication_object_2_async_async_wait PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(communication_object_2_async_async_wait MPI::MPI_CXX GridTools::gridtools Boost::mpi)
add_test(
    NAME communication_object_2_async_async_wait
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} communication_object_2_async_async_wait ${MPIEXEC_POSTFLAGS}
)

add_executable(atlas_integration atlas_integration.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
target_include_directories(atlas_integration PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(atlas_integration MPI::MPI_CXX Boost::mpi GridTools::gridtools atlas)
add_test(
    NAME atlas_integration.cpp
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} atlas_integration ${MPIEXEC_POSTFLAGS}
)


set(_tests_gt data_store_test)

foreach (_t ${_tests_gt})
    add_executable(${_t} ${_t}.cpp ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${_t} MPI::MPI_CXX GridTools::gridtools Boost::mpi)
    add_test(
        NAME ${_t}.cpp
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} ${_t} ${MPIEXEC_POSTFLAGS}
    )
endforeach()
if (USE_GPU)
    foreach (_t ${_tests_gt})
        add_executable(${_t}_gpu ${_t}.cu ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES})
        target_compile_definitions(${_t}_gpu PUBLIC GT_USE_GPU) 
        target_include_directories(${_t}_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
        target_link_libraries(${_t}_gpu MPI::MPI_CXX GridTools::gridtools Boost::mpi)
        add_test(
            NAME ${_t}.cu
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${_ucx_params} ${_t} ${MPIEXEC_POSTFLAGS}
        )
    endforeach()
endif()
