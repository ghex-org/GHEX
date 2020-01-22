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
  integer(8) :: msg_size = 16
  type(ghex_message) :: smsg, rmsg
  integer(1), dimension(:), pointer :: msg_data

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
  !$omp parallel private(thrid, comm, sreq, rreq, smsg, rmsg, msg_data)

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
  msg_data => message_data(smsg)
  msg_data(1:msg_size) = (mpi_rank+1)*10 + thrid;

  ! send / recv with a request, tag 1
  call comm_post_send(comm, smsg, mpi_peer, 1, sreq)
  call comm_post_recv(comm, rmsg, mpi_peer, 1, rreq)

  ! wait for comm
  do while( .not.future_ready(sreq) .or. .not.future_ready(rreq) )
  end do

  ! send / recv with a request, tag 2
  call comm_post_send(comm, smsg, mpi_peer, 2, sreq)
  call comm_post_recv(comm, rmsg, mpi_peer, 2, rreq)

  ! wait for comm
  do while( .not.future_ready(sreq) .or. .not.future_ready(rreq) )
  end do

  ! what have we received?
  msg_data => message_data(rmsg)
  print *, mpi_rank, ": ", thrid, ": ", msg_data

  ! cleanup per-thread
  call message_delete(rmsg)
  call message_delete(smsg)
  call comm_delete(communicators(thrid))

  ! cleanup shared
  !$omp barrier
  !$omp master
  deallocate(communicators)
  !$omp end master

  !$omp end parallel

  ! delete the ghex context
  call context_delete(context)  
  call mpi_finalize(mpi_err)

END PROGRAM test_send_recv_ft
