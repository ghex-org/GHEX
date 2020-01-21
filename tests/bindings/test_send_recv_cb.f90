! problems:
!
! 1. if I use message_new, which creates any_message with 101 constructor,
!    and send that, then inside ghex we have a 'weak' reference (107 constructor),
!    which doesn not have any type info. Hence it cannot free the message.
!    Hence, I cannot use this to 'send and forget', because the buffer will not be deallocated
!
! 2. I can maybe use that for recv resubmission in the callback, but if the real owner
!    should free the memory, then he won't know when it is safe, because he is not aware
!    of the resubmission
!
! 3. request id's returned from cb send/recv variants: the previous request
!    associated with that message is marked as 'completed', but we resubmit.. that
!    request handle is now useless. From the CB we could get a new request, but
!    what do I do with it?

PROGRAM test_send_recv_cb

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
  integer(8) :: msg_size = 16, np
  integer :: recv_completed = 0
  type(ghex_message) :: smsg, rmsg
  type(ghex_request) :: rreq, sreq
  type(ghex_request) :: rtemp
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
  !$omp parallel private(thrid, comm, rreq, sreq, smsg, rmsg, msg_data, np, rtemp)

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

  ! create messages, or get a reference to a shared message
  rmsg = message_new(msg_size, ALLOCATOR_STD)
  smsg = message_new(msg_size, ALLOCATOR_STD)

  ! initialize send data
  msg_data => message_data(smsg)
  msg_data(1:msg_size) = (mpi_rank+1)*10 + thrid;

  !$omp barrier
  ! send / recv with a callback
  call comm_recv_cb(comm, rmsg, mpi_peer, thrid, pcb, rreq)
  call comm_send_cb(comm, smsg, mpi_peer, thrid, req=sreq)

  !$omp barrier
  ! progress the communication
  do while(.not.request_test(sreq) .or. recv_completed /= nthreads)
     np = comm_progress(comm)
  end do
  print *, "request state after", request_test(sreq), request_test(rreq)

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
    type(ghex_request) :: req
    integer(c_int), value :: rank, tag
    integer :: thrid
    integer(1), dimension(:), pointer :: msg_data
    procedure(f_callback), pointer :: pcb
    pcb => recv_callback

    thrid = omp_get_thread_num()+1

    ! what have we received?
    msg_data => message_data(mesg)
    print *, mpi_rank, ": ", thrid, ": ", msg_data

    ! TODO: this should be atomic
    recv_completed = recv_completed + 1

    ! resubmit if needed
    comm = communicators(thrid)
    call comm_resubmit_recv(comm, mesg, rank, tag, pcb, req)
    print *, "recv request has been resubmitted"

  end subroutine recv_callback

END PROGRAM test_send_recv_cb
