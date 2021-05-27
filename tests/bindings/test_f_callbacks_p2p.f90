PROGRAM test_bindings

  use iso_c_binding
  use ghex_message_mod
  use ghex_comm_mod

  include 'mpif.h'  

  integer(c_size_t), parameter :: buffer_size = 1024
  
  integer :: mpi_rank, mpi_size, mpi_mode, mpi_err

  type(ghex_shared_message) :: msg
  type(ghex_communicator) :: comm
  
  integer(1), pointer, dimension(:) :: buffer
  integer :: use_count, ready_comm = 0, n
  procedure(f_callback), pointer :: pcb

  ! init the MPI stuff
  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_mode, mpi_err)
  call mpi_comm_size (mpi_comm_world, mpi_size, mpi_err)
  call mpi_comm_rank (mpi_comm_world, mpi_rank, mpi_err)
  print *, "rank ", mpi_rank, "/", mpi_size
  
  ! create a communicator object
  comm = comm_new();

  ! create a message
  msg = shared_message_new(buffer_size, ALLOCATOR_PERSISTENT_STD)

  ! check the use count (should be 1 at this point - we own it)
  use_count = shared_message_use_count(msg)
  print *, "message use_count 1 == ", use_count

  ! get the data pointer
  buffer => shared_message_data(msg)

  ! fill in the sender data, initialize the receiver memory
  if (mpi_rank == 0) then
    buffer(:) = int([(i, i=1,buffer_size)], kind=1)
  else
    buffer(:) = 0
  end if

  ! define the size of the message
  call shared_message_resize(msg, buffer_size)
  
  ! send/recv the message
  if (mpi_rank == 0) then
    pcb => send_callback
    call comm_send_cb(comm, msg, 1, 1, pcb);
  else
    pcb => recv_callback
    call comm_recv_cb(comm, msg, 0, 1, pcb);
  end if
  
  ! check the use count (should be 2 at this point - scheduled for comm)
  use_count = shared_message_use_count(msg)
  print *, "message use_count 2 == ", use_count

  ! You can free the message as soon as a request is posted.
  ! The message is alive inside GHEX until the comm request is processed.
  ! Note: buffer obtained above is no longer guaranteed to be valid past this point.
  call shared_message_delete(msg)
 
  ! progress the communication
  do while (ready_comm == 0)
    n = comm_progress(comm)
  end do

  ! cleanup
  call comm_delete(comm)
  
  ! finalize
  call mpi_barrier (mpi_comm_world, mpi_err)
  call mpi_finalize (mpi_err)
  
contains

  subroutine send_callback (rank, tag, mesg) bind(c)
    use iso_c_binding
    integer(c_int), value :: rank, tag
    type(ghex_shared_message), value :: mesg
    integer :: use_count

    ! check the use count (should be 3 at this point)
    use_count = shared_message_use_count(mesg)
    print *, "send_callback use count 2 == ", use_count
    
    ready_comm = 1
  end subroutine send_callback

  subroutine recv_callback (rank, tag, mesg) bind(c)
    use iso_c_binding
    integer(c_int), value :: rank, tag
    type(ghex_shared_message), value :: mesg
    integer :: use_count = -10
    
    ! check the use count (should be 3 at this point)
    use_count = shared_message_use_count(mesg)
    print *, "recv_callback use count 2 == ", use_count

    ready_comm = 1
  end subroutine recv_callback

END PROGRAM test_bindings
