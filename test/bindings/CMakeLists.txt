if (GHEX_BUILD_FORTRAN)
    add_subdirectory(fhex)
endif()
#find_package(OpenMP REQUIRED)
#set(_tests test_f_context test_f_communicator test_f_message test_f_barrier test_f_send_recv_ft test_f_send_multi test_f_send_recv_cb)
#
#foreach (_t ${_tests})
#  add_executable(${_t} ${_t}.f90)
#  target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/bindings/fhex)
#  target_link_libraries(${_t} PRIVATE fhex OpenMP::OpenMP_Fortran)
#    add_test(
#        NAME ${_t}.fort
#        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
#    )
#endforeach ()
#
#set(_tests_np6 test_f_cubed_sphere_os)
#
#foreach (_t ${_tests_np6})
#  add_executable(${_t} ${_t}.f90)
#  target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/bindings/fhex)
#  target_link_libraries(${_t} PRIVATE fhex)
#    add_test(
#        NAME ${_t}.fort
#        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
#    )
#endforeach ()
#
##problem with this test TODO fix segfault
#set(_tests_np24 test_f_cubed_sphere)
#foreach (_t ${_tests_np24})
#  add_executable(${_t} ${_t}.f90)
#  target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/bindings/fhex)
#  target_link_libraries(${_t} PRIVATE fhex)
#    add_test(
#        NAME ${_t}.fort
#        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 24 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> ${MPIEXEC_POSTFLAGS}
#    )
#endforeach ()
#
#set(_tests_bench test_f_structured)
#
#foreach (_t ${_tests_bench})
#  add_executable(${_t} ${_t}.f90)
#  target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/bindings/fhex)
#  target_link_libraries(${_t} PRIVATE fhex)
#    add_test(
#        NAME ${_t}.fort
#        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_t}> 64 10 1 8 ${MPIEXEC_POSTFLAGS}
#    )
#endforeach ()
