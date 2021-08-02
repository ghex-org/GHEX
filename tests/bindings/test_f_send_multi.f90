PROGRAM test_send_multi

  ! This test starts m MPI ranks, each with n threads. Each thread sends the same message
  ! to all other ranks (using send_multi variant). So each rank sends n messages.
  ! Each rank pre-posts a single recv request to accept messages from all other ranks.
  ! After completion of a recv request, the receiving thread (i.e. thread, which calls
  ! the recv callback) re-submits the recv request.
  ! The test stops when each rank completes recv of n messages from each other rank.
  ! Then all the outstanding recv requests are canceled.

  use iso_fortran_env
  use omp_lib
  use ghex_mod
  use ghex_comm_mod
  use ghex_message_mod
  use ghex_request_mod

  implicit none

  include 'mpif.h'

  integer :: mpi_err
  integer :: mpi_threading, mpi_size, mpi_rank, mpi_peer
  integer :: nthreads = 0
  integer(8) :: msg_size = 16

  ! shared array to store per-thread communicators (size 0-nthreads-1)
  type(ghex_communicator), dimension(:), pointer :: communicators

  ! recv data structures (size 0-mpi_size-1):
  !  - recv requests to be able to cancel outstanding comm
  !  - shared array to count messages received from each peer rank
  !  - an array of recv messages
  type(ghex_request), dimension(:), pointer :: rrequests
  integer, volatile,  dimension(:), pointer :: rank_received
  type(ghex_message), dimension(:), pointer :: rmsg
  type(ghex_cb_user_data) :: user_data
 
  ! thread-private data
  integer :: it, thrid, peer
  integer, dimension(:), pointer :: peers  ! MPI peer ranks to which to send a message
  type(ghex_communicator) :: comm
  logical :: status

  ! the sent message
  type(ghex_message) :: smsg
  type(ghex_progress_status) :: ps
  integer(1), dimension(:), pointer :: msg_data
  type(ghex_future_multi) :: sfut
  type(ghex_request_multi) :: sreq
  type(ghex_request) :: rreq

  ! recv callback
  procedure(f_callback), pointer :: pcb
  pcb => recv_callback

  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)
  if (MPI_THREAD_MULTIPLE /= mpi_threading) then
    error stop "MPI does not support multithreading"
  end if
  call mpi_comm_size (mpi_comm_world, mpi_size, mpi_err)
  call mpi_comm_rank (mpi_comm_world, mpi_rank, mpi_err)

  ! not (yet) a multi-threaded benchmark
  !$omp parallel
  nthreads = omp_get_num_threads()
  !$omp end parallel

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world);

  ! allocate shared data structures. things related to recv messages
  ! could be allocated here, but have to wait til per-thread communicator
  ! is created below.
  allocate(communicators(0:nthreads-1))

  !$omp parallel private(it, thrid, peer, peers, comm, status, smsg, ps, msg_data, rreq, sfut, sreq)

  ! allocate a communicator per thread and store in a shared array
  thrid = omp_get_thread_num()
  communicators(thrid) = ghex_comm_new()
  comm = communicators(thrid)

  ! these are recv requests - one per peer (mpi_size-1).
  ! could be done outside of the parallel block, but comm is required
  !$omp master

  ! initialize shared datastructures
  allocate(rrequests(0:mpi_size-1))
  allocate(rank_received(0:mpi_size-1))
  rank_received = 0

  ! pre-post a recv
  allocate (rmsg(0:mpi_size-1))
  it = 0
  user_data%data = c_loc(rank_received)
  do while (it<mpi_size)
    if (it/=mpi_rank) then
      rmsg(it) = ghex_message_new(msg_size, GhexAllocatorHost)
      call ghex_comm_recv_cb(comm, rmsg(it), it, it, pcb, user_data=user_data, req=rreq)

      ! NOTE: we have to use a local rreq variable, because rrequests(it)
      ! can be overwritten in the cb routine. cb can be called below in ghex_comm_barrier.
      ! As a result, rrequests(it) is overwritten with a completed request
      ! by the above ghex_comm_recv_cb call.
      if (.not.ghex_request_test(rreq)) then
        rrequests(it) = rreq
      end if
    end if
    it = it+1
  end do
  !$omp end master

  ! wait for master to init the arrays
  call ghex_comm_barrier(comm, GhexBarrierThread)

  ! create list of peers (exclude self)
  allocate (peers(1:mpi_size-1))
  it   = 0
  peer = 1
  do while(it < mpi_size)
    if (it /= mpi_rank) then
      peers(peer) = it
      peer = peer+1
    end if
    it = it+1
  end do

  ! initialize send data
  smsg = ghex_message_new(msg_size, GhexAllocatorHost)
  msg_data => ghex_message_data(smsg)
  msg_data(1:msg_size) = (mpi_rank+1)*nthreads + thrid;

  ! send without a callback (returns a future), keep ownership of the message
  call ghex_comm_post_send_multi(comm, smsg, peers, mpi_rank, future=sfut)

  ! progress the communication - complete the send before posting another one
  call ghex_future_wait(sfut)

  ! send with a callback (can be empty), keep ownership of the message
  call ghex_comm_post_send_multi_cb(comm, smsg, peers, mpi_rank, req=sreq)

  ! progress the communication - complete the send before posting another one
  do while(.not.ghex_request_test(sreq))
     ps = ghex_comm_progress(comm)
  end do

  ! send with a callback (can be empty), give ownership of the message to ghex: smsg buffer will be freed after completion
  call ghex_comm_send_multi_cb(comm, smsg, peers, mpi_rank)

  ! wait for all recv requests to complete - enough if only master does this,
  ! the other threads also have to progress the communication, but that happens
  ! in the call to ghex_comm_barrier below
  !$omp master
  it   = 0
  do while(it < mpi_size)
    if (it /= mpi_rank) then
      do while (rank_received(it) /= nthreads*3)
        ps = ghex_comm_progress(comm)
      end do
      print *, mpi_rank, "received", rank_received(it), "messages from rank", it
    end if
    it = it+1
  end do
  !$omp end master

  ! wait for all threads and ranks to complete the recv.
  ! ghex_comm_barrier is safe as it progresses all communication: MPI and GHEX
  call ghex_comm_barrier(comm, GhexBarrierThread)

  ! cancel all outstanding recv requests
  !$omp master
  it   = 0
  do while(it < mpi_size)
    if (it /= mpi_rank) then
      status = ghex_request_cancel(rrequests(it))
      if (.not.status) then
        print *, "failed to cancel a recv request"
        call exit(1)
      else
        print *, "SUCCEEDED to cancel a recv request"
      end if
    end if
    it = it+1
  end do
  !$omp end master

  ! cleanup per-thread. messages are freed by ghex.
  deallocate (peers)
  call ghex_free(comm)

  !$omp end parallel

  ! cleanup shared
  deallocate (rmsg)
  deallocate (communicators)
  deallocate (rrequests)
  deallocate (rank_received)

  call ghex_finalize()
  call mpi_finalize(mpi_err)

CONTAINS

  subroutine recv_callback (mesg, rank, tag, user_data) bind(c)
    use iso_c_binding
    type(ghex_message), value :: mesg
    integer(c_int), value :: rank, tag
    type(ghex_cb_user_data), value :: user_data
    
    type(ghex_request) :: rreq
    integer :: thrid
    integer, volatile, dimension(:), pointer :: rank_received

    ! NOTE: this segfaults in Intel compiler. It seems we have to use
    ! the globally defined pcb from the main function. WHY??
    ! procedure(f_callback), pointer :: pcb
    ! pcb => recv_callback

    ! needed to know which communicator we can use. Communicators are bound to threads.
    thrid = omp_get_thread_num()

    ! mark receipt in the user data
    ! OBS: this array is now 1-based, not 0-based as the original
    call c_f_pointer(user_data%data, rank_received, [mpi_size])

    !$omp atomic
    rank_received(tag+1) = rank_received(tag+1) + 1
    !$omp end atomic

    ! resubmit
    call ghex_comm_resubmit_recv(communicators(thrid), mesg, rank, tag, pcb, rreq, user_data=user_data)

    ! only store non-completed requests
    if (.not.ghex_request_test(rreq)) then
       rrequests(tag) = rreq
    end if

  end subroutine recv_callback

END PROGRAM test_send_multi
