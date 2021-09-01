PROGRAM test_send_recv_ft
  use omp_lib
  use ghex_mod
  use ghex_comm_mod
  use ghex_message_mod
  use ghex_future_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err
  integer :: mpi_threading, mpi_size, mpi_rank, mpi_peer
  integer :: nthreads = 0, thrid

  type(ghex_communicator) :: comm
  type(ghex_future) :: sreq, rreq

  ! message
  integer(8) :: msg_size = 16
  type(ghex_message) :: smsg, rmsg
  integer(1), dimension(:), pointer :: msg_data

  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)
  if (MPI_THREAD_MULTIPLE /= mpi_threading) then
    stop "MPI does not support multithreading"
  end if
  call mpi_comm_size (mpi_comm_world, mpi_size, mpi_err)
  call mpi_comm_rank (mpi_comm_world, mpi_rank, mpi_err)
  if (mpi_size /= 2) then
     if (mpi_rank == 0) then
        print *, "Usage: this test can only be executed for 2 ranks"
     end if
     call mpi_finalize(mpi_err)
     call exit(1)
  end if
  mpi_peer = modulo(mpi_rank+1, 2)

  !$omp parallel shared(nthreads)
  nthreads = omp_get_num_threads()
  !$omp end parallel

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world);

  !$omp parallel private(thrid, comm, sreq, rreq, smsg, rmsg, msg_data)

  ! make thread id 1-based
  thrid = omp_get_thread_num()+1

  ! allocate a communicator per thread
  comm = ghex_comm_new()

  ! create a message per thread
  rmsg = ghex_message_new(msg_size, GhexAllocatorHost)
  smsg = ghex_message_new(msg_size, GhexAllocatorHost)
  msg_data => ghex_message_data(smsg)
  msg_data(1:msg_size) = (mpi_rank+1)*nthreads + thrid;

  ! send / recv with a request, tag 1
  call ghex_comm_post_send(comm, smsg, mpi_peer, thrid, sreq)
  call ghex_comm_post_recv(comm, rmsg, mpi_peer, thrid, rreq)

  ! wait for comm
  do while( .not.ghex_future_ready(sreq) .or. .not.ghex_future_ready(rreq) )
  end do

  ! what have we received?
  msg_data => ghex_message_data(rmsg)
  if (any(msg_data /= (mpi_peer+1)*nthreads + thrid)) then
    print *, "wrong data received"
    print *, mpi_rank, ": ", thrid, ": ", msg_data
    call exit(1)
  end if

  ! cleanup per-thread
  call ghex_free(rmsg)
  call ghex_free(smsg)
  call ghex_free(comm)

  !$omp end parallel

  call ghex_finalize()  
  call mpi_finalize(mpi_err)

END PROGRAM test_send_recv_ft
