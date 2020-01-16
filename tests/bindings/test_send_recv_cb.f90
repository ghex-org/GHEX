PROGRAM test_send_recv_ft

  use omp_lib
  use ghex_context_mod
  use ghex_comm_mod
  use ghex_message_mod
  use ghex_future_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err
  integer :: mpi_threading, mpi_size, mpi_rank, mpi_peer
  integer :: nthreads = 0, thrid

  ! comm context
  type(ghex_context) :: context
  type(ghex_communicator), dimension(:), pointer :: communicators
  type(ghex_communicator) :: comm
  type(ghex_future) :: sreq, rreq

  ! message
  integer(8) :: msg_size = 16, np
  type(ghex_message) :: smsg, rmsg
  integer(1), dimension(:), pointer :: msg_data
  
  procedure(f_callback), pointer :: pcb
  pcb => recv_callback

  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)
  call mpi_comm_size (mpi_comm_world, mpi_size, mpi_err)
  call mpi_comm_rank (mpi_comm_world, mpi_rank, mpi_err)
  if (mpi_size /= 2) then
     if (mpi_rank == 0) then
        print *, "Usage: this test can only be executed for 2 ranks"
     end if
     call mpi_finalize(mpi_err)
     call exit(0)
  end if
  mpi_peer = modulo(mpi_rank+1, 2)

  !$omp parallel shared(nthreads)
  nthreads = omp_get_num_threads()
  !$omp end parallel

  ! create a context object
  context = context_new(nthreads, mpi_comm_world);

  ! make per-thread communicators
  !$omp parallel private(thrid, comm, sreq, rreq, smsg, rmsg, msg_data, np)

  ! make thread id 1-based
  thrid = omp_get_thread_num()+1

  ! initialize shared datastructures
  !$omp master
  allocate(communicators(nthreads))  
  !$omp end master
  !$omp barrier

  ! allocate a communicator per thread and store in a shared array
  communicators(thrid) = context_get_communicator(context)
  comm = communicators(thrid)

  ! create a message per thread
  rmsg = message_new(msg_size, ALLOCATOR_STD)
  smsg = message_new(msg_size, ALLOCATOR_STD)

  ! initialize send data
  msg_data => message_data(smsg)
  msg_data(1:msg_size) = (mpi_rank+1)*10 + thrid;
  
  ! send / recv with a callback
  call comm_send_cb(comm, smsg, mpi_peer, 1)
  call comm_recv_cb(comm, rmsg, mpi_peer, 1, pcb)

  ! print *, "msg use count", message_use_count(smsg), message_use_count(rmsg)

  do while(message_use_count(smsg)>1 .or. message_use_count(rmsg)>1)
     np = comm_progress(comm)
  end do

  ! what have we received?
  msg_data => message_data(rmsg)
  print *, mpi_rank, ": ", thrid, ": ", msg_data

  ! cleanup per-thread
  call message_delete(smsg)
  call message_delete(rmsg)
  call comm_delete(communicators(thrid))

  ! cleanup shared
  !$omp barrier
  !$omp master
  deallocate(communicators)
  !$omp end master

  !$omp end parallel

  ! delete the ghex context
  call context_delete(context)  

contains

  subroutine recv_callback (mesg, rank, tag) bind(c)
    use iso_c_binding
    type(ghex_message), value :: mesg
    integer(c_int), value :: rank, tag
    integer :: use_count = -1

    ! check the use count (should be 3 at this point)
    use_count = message_use_count(mesg)
    ! print *, "callback use count ", use_count
  end subroutine recv_callback

END PROGRAM test_send_recv_ft
