PROGRAM fhex_bench
  use iso_fortran_env
#ifdef GHEX_USE_OPENMP
  use omp_lib
#endif
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
#ifdef GHEX_USE_OPENMP
  !$omp threadprivate(comm_cnt, nlsend_cnt, nlrecv_cnt, submit_cnt, submit_recv_cnt, thread_id)
#endif

#ifdef GHEX_USE_OPENMP
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

#ifdef GHEX_USE_OPENMP
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

#ifdef GHEX_USE_OPENMP
  !$omp parallel
#endif

  call run()

#ifdef GHEX_USE_OPENMP
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
    integer :: i = 0, last_i = 0, j = 0
    integer :: incomplete_sends = 0, send_complete = 0
    type(ghex_progress_status), save :: np

    type(ghex_message), dimension(:), allocatable, save :: smsgs, rmsgs
    type(ghex_request), dimension(:), allocatable, save :: sreqs, rreqs
    type(ghex_future), save :: bsreq, brreq
    type(ghex_message), save :: bsmsg, brmsg
    logical :: result = .false.
    real :: ttic = 0, tic = 0, toc = 0
  
    procedure(f_callback), pointer, save :: rcb, scb

    !$omp threadprivate(comm, rank, size, num_threads, peer_rank)
    !$omp threadprivate(using_mt, last_received, last_sent)
    !$omp threadprivate(dbg, sdbg, rdbg, i, last_i, j, np, incomplete_sends, send_complete)
    !$omp threadprivate(smsgs, rmsgs, sreqs, rreqs)
    !$omp threadprivate(bsreq, brreq, bsmsg, brmsg, result, rcb, scb)
    !$omp threadprivate(ttic, tic, toc)
   
    ! ---------------------------------------
    ! world info
    ! ---------------------------------------

    ! obtain a communicator
    comm = ghex_comm_new()

    rank        = ghex_comm_rank(comm);
    size        = ghex_comm_size(comm);
#ifdef GHEX_USE_OPENMP
    thread_id   = omp_get_thread_num()
    num_threads = omp_get_num_threads()
#else
    thread_id   = 0
    num_threads = 1
#endif
    peer_rank   = modulo(rank+1, 2)

#ifdef GHEX_USE_OPENMP
    using_mt = .true.
#endif

    if (thread_id==0 .and. rank==0) then
       call getarg(0, pname)
       print *, "running ", pname
    end if

    ! ---------------------------------------
    ! data initialization
    ! ---------------------------------------
    rcb => recv_callback
    scb => send_callback

    allocate(smsgs(inflight), rmsgs(inflight), sreqs(inflight), rreqs(inflight))
    do j = 1, inflight
       smsgs(j) = ghex_message_new(buff_size, GhexAllocatorHost);
       rmsgs(j) = ghex_message_new(buff_size, GhexAllocatorHost);
       call ghex_message_zero(smsgs(j))
       call ghex_message_zero(rmsgs(j))
       call ghex_request_init(sreqs(j))
       call ghex_request_init(rreqs(j))
    end do

    call ghex_comm_barrier(comm, GhexBarrierGlobal)

    if (thread_id == 0) then
       call cpu_time(ttic)
       tic = ttic
       if(rank == 1) then
          print *, "number of threads: ", num_threads, ", multi-threaded: ", using_mt
       end if
    end if

    ! ---------------------------------------   
    ! send / recv niter messages, work in inflight requests at a time
    ! ---------------------------------------   
    do while(i < niter)

       call ghex_comm_barrier(comm, GhexBarrierThread)
       
       if (thread_id == 0 .and. dbg >= (niter/10)) then
          dbg = 0
          call cpu_time(toc)
          print *, rank, " total bwdt MB/s:      ", &
               (i-last_i)*size*buff_size/(toc-tic)*num_threads/1e6
          tic = toc
          last_i = i;
       end if

       ! submit inflight requests
       do j = 1, inflight
          dbg = dbg + num_threads
          i = i + num_threads
          call ghex_comm_post_recv_cb(comm, rmsgs(j), peer_rank, thread_id*inflight+j-1, rcb, rreqs(j))
          call ghex_comm_post_send_cb(comm, smsgs(j), peer_rank, thread_id*inflight+j-1, scb, sreqs(j))
       end do

       ! complete all inflight requests before moving on
       do while (sent < num_threads*inflight .or. received < num_threads*inflight)
          np = ghex_comm_progress(comm)
       end do       

       call ghex_comm_barrier(comm, GhexBarrierThread)
       
       sent = 0
       received = 0
    end do

    call ghex_comm_barrier(comm, GhexBarrierGlobal)

    ! ---------------------------------------
    ! Timing and statistics output
    ! ---------------------------------------
    if (thread_id==0 .and. rank == 0) then
       call cpu_time(toc)
       print *, "time:", (toc-ttic)/num_threads
       print *, "final MB/s: ", (niter*size*buff_size)/(toc-ttic)*num_threads/1e6
    end if


    ! stop here to help produce a nice std output
    call ghex_comm_barrier(comm, GhexBarrierGlobal)
#ifdef GHEX_USE_OPENMP
    !$omp critical
#endif
101 format ("rank ", I0, " thread " , I0 , " sends submitted " , I0,  &
          " serviced " , I0 , ", non-local sends " ,  I0 , " non-local recvs " , I0)
    write (*, 101) rank, thread_id , submit_cnt/num_threads , comm_cnt, nlsend_cnt , nlrecv_cnt
#ifdef GHEX_USE_OPENMP
    !$omp end critical
#endif    

    ! tail loops - not needed in wait benchmarks

    ! ---------------------------------------
    ! cleanup
    ! ---------------------------------------
    do j = 1, inflight
       call ghex_free(smsgs(j))
       call ghex_free(rmsgs(j))
    end do
    deallocate(smsgs, rmsgs, sreqs, rreqs)

    call ghex_free(comm)
  end subroutine run

  ! ---------------------------------------
  ! callbacks
  ! ---------------------------------------
  subroutine send_callback (mesg, rank, tag, user_data)
    type(ghex_message), value :: mesg
    integer(c_int), value :: rank, tag
    integer(1), dimension(:), pointer, save :: msg_data
    type(ghex_cb_user_data), value :: user_data

    if(tag/inflight /= thread_id) nlsend_cnt = nlsend_cnt + 1;
    comm_cnt = comm_cnt + 1;
    call atomic_add(sent, 1);
  end subroutine send_callback

  subroutine recv_callback (mesg, rank, tag, user_data)
    type(ghex_message), value :: mesg
    integer(c_int), value :: rank, tag
    type(ghex_cb_user_data), value :: user_data
    
    if(tag/inflight /= thread_id) nlrecv_cnt = nlrecv_cnt + 1;
    comm_cnt = comm_cnt + 1;
    call atomic_add(received, 1);
  end subroutine recv_callback

#ifndef GHEX_USE_OPENMP
subroutine atomic_add(var, val)
  integer :: var, val
  var = var + val
end subroutine atomic_add
#endif

END PROGRAM fhex_bench
