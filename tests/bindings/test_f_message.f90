PROGRAM test_message
  use ghex_mod
  use ghex_message_mod

  implicit none  

  include 'mpif.h'  

  integer(8) :: msg_size = 16, i, mpi_err, mpi_mode
  type(ghex_message) :: msg1, msg2
  integer(1), dimension(:), pointer :: msg_data

  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_mode, mpi_err)
  call ghex_init(1, mpi_comm_world)
  
  msg1 = ghex_message_new(msg_size, GhexAllocatorHost)
  msg2 = ghex_message_new(msg_size, GhexAllocatorHost)
  
  msg_data => ghex_message_data(msg1)
  msg_data(1:msg_size) = (/(i, i=1,msg_size,1)/)

  print *, "values:    ", msg_data

  ! cleanup
  call ghex_free(msg1)
  call ghex_free(msg2)
  call ghex_finalize()
  call mpi_finalize (mpi_err)

END PROGRAM test_message
