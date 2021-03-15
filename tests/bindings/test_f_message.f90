PROGRAM test_message
  use mpi
  use ghex_mod
  use ghex_message_mod

  implicit none  

  integer(8) :: msg_size = 16, i
  type(ghex_message) :: msg
  integer(1), dimension(:), pointer :: msg_data
  
  msg = ghex_message_new(msg_size, ALLOCATOR_STD)
  
  msg_data => ghex_message_data(msg)
  msg_data(1:msg_size) = (/(i, i=1,msg_size,1)/)

  print *, "values:    ", msg_data

  ! cleanup
  call ghex_free(msg)

END PROGRAM test_message
