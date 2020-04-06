PROGRAM fhex_bench
  use iso_fortran_env
  use omp_lib
  use ghex_mod
  use ghex_comm_mod
  use ghex_message_mod
  use ghex_request_mod
  use ghex_future_mod
  
  implicit none  
  
  include 'mpif.h'  

  ! threadprivate variables
  integer :: comm_cnt = 0, nlsend_cnt = 0, nlrecv_cnt = 0, submit_cnt = 0, submit_recv_cnt = 0
  integer :: thread_id = 0
#ifdef USE_OPENMP
  !$omp threadprivate(comm_cnt, nlsend_cnt, nlrecv_cnt, submit_cnt, submit_recv_cnt, thread_id)
#endif

#ifdef USE_OPENMP
  integer(atomic_int_kind) :: sent[*] = 0, received[*] = 0, tail_send[*] = 0, tail_recv[*] = 0
#else
  integer :: sent = 0, received = 0, tail_send = 0, tail_recv = 0
#endif

  ! local variables
  integer :: mpi_err, mpi_threading
  integer :: num_threads = 1
  character(len=32) :: arg
  integer(8) :: niter, buff_size
  integer :: inflight
  
  if( iargc() /= 3) then
     print *, "Usage: bench [niter] [msg_size] [inflight]";
     call exit(1)
  end if

  call getarg(1, arg);
  read(arg,*) niter
  call getarg(2, arg);
  read(arg,*) buff_size
  call getarg(3, arg);
  read(arg,*) inflight

#ifdef USE_OPENMP
  !$omp parallel
  num_threads = omp_get_num_threads()
  !$omp end parallel
  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)
  if (mpi_threading /= MPI_THREAD_MULTIPLE) then
     print *, "MPI_THREAD_MULTIPLE not supported by MPI, aborting";
     call exit(1)
  end if
#else
  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
#endif

  ! init ghex
  call ghex_init(num_threads, mpi_comm_world);

#ifdef USE_OPENMP
  !$omp parallel
#endif

  call run()

#ifdef USE_OPENMP
  !$omp end parallel
#endif

  call ghex_finalize()  
  call mpi_finalize(mpi_err)

contains

  subroutine run()

    character(len=256) :: pname
    
    ! all below variables are thread-local
    type(ghex_communicator), save :: comm

    integer :: rank = -1, size = -1, num_threads = -1, peer_rank = -1
    logical :: using_mt = .false.

    integer :: last_received = 0
    integer :: last_sent = 0
    integer :: dbg = 0, sdbg = 0, rdbg = 0;
    integer :: j = 0
    integer :: incomplete_sends = 0, send_complete = 0
    type(ghex_progress_status), save :: np

    type(ghex_message), dimension(:), allocatable, save :: smsgs, rmsgs
    type(ghex_future),  dimension(:), allocatable, save :: sreqs, rreqs
    type(ghex_future),  save :: bsreq, brreq
    type(ghex_message), save :: bsmsg, brmsg
    logical :: result = .false.
    real :: ttic = 0, tic = 0, toc = 0
  
    !$omp threadprivate(comm, rank, size, num_threads, peer_rank)
    !$omp threadprivate(using_mt, last_received, last_sent)
    !$omp threadprivate(dbg, sdbg, rdbg, j, np, incomplete_sends, send_complete)
    !$omp threadprivate(smsgs, rmsgs, sreqs, rreqs)
    !$omp threadprivate(bsreq, brreq, bsmsg, brmsg, result)
    !$omp threadprivate(ttic, tic, toc)
   
    ! ---------------------------------------
    ! world info
    ! ---------------------------------------

    ! obtain a communicator
    comm = ghex_comm_new()

    rank        = ghex_comm_rank(comm);
    size        = ghex_comm_size(comm);
    thread_id   = omp_get_thread_num();
    num_threads = omp_get_num_threads();
    peer_rank   = modulo(rank+1, 2)

#ifdef USE_OPENMP
    using_mt = .true.
#endif

    if (thread_id==0 .and. rank==0) then
       call getarg(0, pname)
       print *, "running ", pname
    end if

    ! ---------------------------------------
    ! data initialization
    ! ---------------------------------------
    allocate(smsgs(inflight), rmsgs(inflight), sreqs(inflight), rreqs(inflight))
    do j = 1, inflight
       smsgs(j) = ghex_message_new(buff_size, ALLOCATOR_STD);
       rmsgs(j) = ghex_message_new(buff_size, ALLOCATOR_STD);
       call ghex_message_zero(smsgs(j))
       call ghex_message_zero(rmsgs(j))
       call ghex_future_init(sreqs(j))
       call ghex_future_init(rreqs(j))
    end do

    call ghex_comm_barrier(comm)

    if (thread_id == 0) then
       call cpu_time(ttic)
       tic = ttic
       if(rank == 1) then
          print *, "number of threads: ", num_threads, ", multi-threaded: ", using_mt
       end if
    end if

    ! pre-post comm
    do j = 1, inflight
       call ghex_comm_post_recv(comm, rmsgs(j), peer_rank, thread_id*inflight+j-1, rreqs(j))
       call ghex_comm_post_send(comm, smsgs(j), peer_rank, thread_id*inflight+j-1, sreqs(j))
    end do

    ! ---------------------------------------
    ! send/recv niter messages - as soon as a slot becomes free
    ! ---------------------------------------
    do while(sent < niter .or. received < niter)
       if (thread_id == 0 .and. dbg >= (niter/10)) then
          dbg = 0
          call cpu_time(toc)
          print *, rank, " total bwdt MB/s:      ", &
               (received-last_received + sent-last_sent)*size*buff_size/2/(toc-tic)*num_threads/1e6
          tic = toc
          last_received = received;
          last_sent = sent;
       end if

       if (rank==0 .and. thread_id==0 .and. rdbg >= (niter/10)) then
          print *,  received, " received"
          rdbg = 0
       end if

       if (rank==0 .and. thread_id==0 .and. sdbg >= (niter/10)) then
          print *, sent, " sent"
          sdbg = 0
       end if
       
       do j = 1, inflight
          if (ghex_future_ready(rreqs(j))) then
             call atomic_add(received, 1)
             rdbg = rdbg + num_threads
             dbg = dbg + num_threads
             call ghex_comm_post_recv(comm, rmsgs(j), peer_rank, thread_id*inflight+j-1, rreqs(j))
          end if

          if (sent < niter .and. ghex_future_ready(sreqs(j))) then
             call atomic_add(sent, 1)
             sdbg = sdbg + num_threads
             dbg = dbg + num_threads
             call ghex_comm_post_send(comm, smsgs(j), peer_rank, thread_id*inflight+j-1, sreqs(j))
          end if
       end do

       ! j = ghex_future_test_any(rreqs)
       ! if (j <= inflight) then
       !    call atomic_add(received, 1)
       !    rdbg = rdbg + num_threads
       !    dbg = dbg + num_threads
       !    call ghex_comm_post_recv(comm, rmsgs(j), peer_rank, thread_id*inflight+j-1, rreqs(j))
       ! end if

       ! if (sent < niter) then
       !    j = ghex_future_test_any(sreqs)
       !    if (j <= inflight) then
       !       call atomic_add(sent, 1)
       !       sdbg = sdbg + num_threads
       !       dbg = dbg + num_threads
       !       call ghex_comm_post_send(comm, smsgs(j), peer_rank, thread_id*inflight+j-1, sreqs(j))
       !    end if
       ! end if

    end do

    call ghex_comm_barrier(comm)

    ! ---------------------------------------
    ! Timing and statistics output
    ! ---------------------------------------
    if (thread_id==0 .and. rank == 0) then
       call cpu_time(toc)
       print *, "time:", (toc-ttic)/num_threads
       print *, "final MB/s: ", (niter*size*buff_size)/(toc-ttic)*num_threads/1e6
    end if

    ! ---------------------------------------
    ! tail loops - submit RECV requests until
    ! all SEND requests have been finalized.
    ! This is because UCX cannot cancel SEND requests.
    ! https://github.com/openucx/ucx/issues/1162
    ! ---------------------------------------

    ! complete all posted sends
    do while(tail_send/=num_threads)

       np = ghex_comm_progress(comm)

       ! check if we have completed all our posted sends
       if(send_complete == 0) then
          incomplete_sends = 0;
          do j = 1, inflight
             if(.not. ghex_future_ready(sreqs(j))) then
                incomplete_sends = incomplete_sends + 1
             end if
          end do

          if (incomplete_sends == 0) then
             ! increase thread counter of threads that are done with the sends
             call atomic_add(tail_send, 1)
             send_complete = 1
          end if
       end if

       ! continue to re-schedule all recvs to allow the peer to complete
       do j = 1, inflight
          if (ghex_future_ready(rreqs(j))) then
             call ghex_comm_post_recv(comm, rmsgs(j), peer_rank, thread_id*inflight+j-1, rreqs(j))
          end if
       end do
    end do

    ! We have all completed the sends, but the peer might not have yet.
    ! Notify the peer and keep submitting recvs until we get his notification.

#if USE_OPENMP
    !$omp master
#endif
    bsmsg = ghex_message_new(1_8, ALLOCATOR_STD);
    brmsg = ghex_message_new(1_8, ALLOCATOR_STD);
    call ghex_comm_post_send(comm, bsmsg, peer_rank, 800000, bsreq);
    call ghex_comm_post_recv(comm, brmsg, peer_rank, 800000, brreq);
#if USE_OPENMP
    !$omp end master
#endif

    do while(tail_recv == 0)

       np = ghex_comm_progress(comm)
       
       ! schedule all recvs to allow the peer to complete
       do j = 1, inflight
          if (ghex_future_ready(rreqs(j))) then
             call ghex_comm_post_recv(comm, rmsgs(j), peer_rank, thread_id*inflight+j-1, rreqs(j))
          end if
       end do

#if USE_OPENMP
       !$omp master
#endif
       if (ghex_future_ready(brreq)) then
          tail_recv = 1
       end if
#if USE_OPENMP
    !$omp end master
#endif
    end do

    ! peer has sent everything, so we can cancel all posted recv requests
    do j = 1, inflight
       result = ghex_future_cancel(rreqs(j))
    end do

    ! ---------------------------------------
    ! cleanup
    ! ---------------------------------------
    do j = 1, inflight
       call ghex_message_delete(smsgs(j))
       call ghex_message_delete(rmsgs(j))
    end do
    deallocate(smsgs, rmsgs, sreqs, rreqs)

    call ghex_comm_delete(comm)

  end subroutine run

#ifndef USE_OPENMP
subroutine atomic_add(var, val)
  integer :: var, val
  var = var + val
end subroutine atomic_add
#endif

END PROGRAM fhex_bench
