PROGRAM test_bindings

  use iso_c_binding
  use ghex_message_mod
  use ghex_comm_mod

  include 'mpif.h'  

  integer(c_size_t), parameter :: buffer_size = 1024
  
  integer :: mpi_rank, mpi_size, mpi_mode, mpi_err
  integer, dimension(:), allocatable :: nbors

  type(ghex_shared_message) :: msg
  type(ghex_communicator) :: comm
  
  integer(1), pointer, dimension(:) :: buffer
  integer :: use_count
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

    ! construct neighbor arrays
    allocate(nbors(1:mpi_size-1))
    nbors(:) = int([(i, i=1,mpi_size-1)], kind=1)

    ! submitt send requests, do not wait for a completion callback
    call comm_send_multi(comm, msg, nbors, 1);
  else

    ! submit recv requests, expect a completion callback 
    pcb => recv_callback
    call comm_recv_cb(comm, msg, 0, 1, pcb);
  end if

  ! check the use count (should be 2 on receivers and mpi_size on sender)
  use_count = shared_message_use_count(msg)
  print *, "message use_count after comm == ", use_count

  ! You can free the message as soon as a request is posted.
  ! The message is alive inside GHEX until the comm request is processed.
  ! Note: buffer obtained above is no longer guaranteed to be valid past this point.
  call shared_message_delete(msg)
  
  ! progress the communication untill no more messages in the queue
  do while (comm_progress(comm) /= 0)
  end do

  ! cleanup
  call comm_delete(comm)
  
  ! finalize
  call mpi_barrier (mpi_comm_world, mpi_err)
  call mpi_finalize (mpi_err)

contains

  subroutine recv_callback (rank, tag, mesg) bind(c)
    use iso_c_binding
    integer(c_int), value :: rank, tag
    type(ghex_shared_message), value :: mesg
    integer :: use_count
    
    use_count = shared_message_use_count(mesg)
    print *, "recv_callback use_count ", use_count
  end subroutine recv_callback

END PROGRAM test_bindings
