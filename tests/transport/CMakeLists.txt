set(_tests test_low_level test_low_level_x test_send_multi test_cancel_request test_attach_detach)
foreach(t_ ${_tests})
  add_executable( ${t_} ./${t_}.cpp )
  target_link_libraries(${t_}  GridTools::gridtools MPI::MPI_CXX gtest_main_mt ghexlib)
  add_test(
    NAME ${t_}.cpp
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${t_} ${MPIEXEC_POSTFLAGS}
  )
endforeach(t_ ${_tests})

