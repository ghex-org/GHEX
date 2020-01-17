PROGRAM test_message

  use omp_lib
  use ghex_message_mod

  implicit none  

  include 'mpif.h'  

  integer(8) :: msg_size = 16, i
  type(ghex_message) :: msg
  type(ghex_shared_message) :: smsg
  integer(1), dimension(:), pointer :: msg_data
  
  smsg = shared_message_new(msg_size, ALLOCATOR_STD)
  msg  = shared_message_ref(smsg)
  
  msg_data => message_data(msg)
  msg_data(1:msg_size) = (/(i, i=1,msg_size,1)/)

  print *, "values:    ", msg_data
  print *, "use count: ", message_use_count(msg)

  ! cleanup
  call message_delete(msg)
  call shared_message_delete(smsg)

END PROGRAM test_message
