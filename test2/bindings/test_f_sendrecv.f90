PROGRAM test_sendrecv

  use iso_c_binding
  use ghex_future_mod
  use ghex_message_mod
  use ghex_comm_mod

  implicit none  
  
  include 'mpif.h'  

  integer(c_size_t), parameter :: buffer_size = 1024
  
  integer :: mpi_rank, mpi_size, mpi_mode, mpi_err

  type(ghex_shared_message) :: msg
  type(ghex_communicator) :: comm
  type(ghex_future) :: future
  
  integer(1), pointer, dimension(:) :: buffer
  integer :: use_count, i
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

    ! submitt send requests
    i = 1
    do while (i <= mpi_size-1)
      future = comm_send(comm, msg, i, 1);
      call future_wait(future)
      i = i + 1
    end do
  else

    ! submit recv requests
    future = comm_recv(comm, msg, 0, 1);
    call future_wait(future)
  end if

  ! check the use count (should be 2 on receivers and mpi_size on sender)
  use_count = shared_message_use_count(msg)
  print *, "message use_count after comm == ", use_count

  call shared_message_delete(msg)

  ! cleanup
  call comm_delete(comm)
  
  ! finalize
  call mpi_barrier (mpi_comm_world, mpi_err)
  call mpi_finalize (mpi_err)

contains

END PROGRAM test_sendrecv
