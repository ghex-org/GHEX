PROGRAM test_message

  use omp_lib
  use ghex_message_mod

  implicit none  

  include 'mpif.h'  

  integer(8) :: msg_size = 16, i
  type(ghex_message) :: msg
  integer(1), dimension(:), pointer :: msg_data
  
  msg = message_new(msg_size, ALLOCATOR_STD)
  
  msg_data => message_data(msg)
  msg_data(1:msg_size) = (/(i, i=1,msg_size,1)/)

  print *, "values:    ", msg_data

  ! cleanup
  call message_delete(msg)

END PROGRAM test_message
