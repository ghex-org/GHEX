PROGRAM test_send_recv_ft

  use omp_lib
  use ghex_context_mod
  use ghex_comm_mod
  use ghex_message_mod
  use ghex_request_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err
  integer :: mpi_threading, mpi_size, mpi_rank, mpi_peer
  integer :: nthreads = 0, thrid

  ! comm context
  type(ghex_context) :: context
  type(ghex_communicator), dimension(:), pointer :: communicators
  type(ghex_communicator) :: comm

  ! message
  integer(8) :: msg_size = 16000000, np
  integer :: recv_completed = 0
  type(ghex_message) :: smsg, rmsg
  type(ghex_request) :: sreq, rreq
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

  ! create messages
  rmsg = message_new(msg_size, ALLOCATOR_STD)
  smsg = message_new(msg_size, ALLOCATOR_STD)

  ! initialize send data
  msg_data => message_data(smsg)
  msg_data(1:msg_size) = (mpi_rank+1)*10 + thrid;
  
  ! send / recv with a callback
  sreq = comm_send_cb(comm, smsg, mpi_peer, 1)
  rreq = comm_recv_cb(comm, rmsg, mpi_peer, 1, pcb)
  
  ! use count is always 2 for recv, can be 1 or 2 for send: send can complete in-place here
  print *, "msg use count", message_use_count(smsg), message_use_count(rmsg)
  print *, "request state before", request_test(sreq), request_test(rreq)

  ! can safely delete the local instance of the shared message:
  ! ownership is passed to ghex, and the message will be returned to us
  ! in the recv callback
  call message_delete(rmsg)

  ! progress the communication
  do while(message_use_count(smsg)>1 .or. recv_completed /= nthreads)
     np = comm_progress(comm)
  end do
  print *, "request state after", request_test(sreq), request_test(rreq)

  ! cleanup per-thread
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

contains

  subroutine recv_callback (mesg, rank, tag) bind(c)
    use iso_c_binding
    type(ghex_message), value :: mesg
    integer(c_int), value :: rank, tag
    integer :: use_count = -1, thrid
    integer(1), dimension(:), pointer :: msg_data

    thrid = omp_get_thread_num()+1

    ! check the use count (should be 1 or 2 at this point)
    use_count = message_use_count(mesg)
    print *, "callback use count ", use_count

    ! what have we received?
    msg_data => message_data(mesg)
    ! print *, mpi_rank, ": ", thrid, ": ", msg_data

    ! TODO: this should be atomic
    recv_completed = recv_completed + 1

    ! TODO: recv data from GHEX
    ! TODO: resubmit: need recursive locks
  end subroutine recv_callback

END PROGRAM test_send_recv_ft
