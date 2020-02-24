PROGRAM test_context
  use omp_lib
  use ghex_mod
  use ghex_comm_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err
  integer :: mpi_threading
  integer :: nthreads = 0, thrid
  integer :: np = 0
  type(ghex_communicator), dimension(:), pointer :: communicators
  type(ghex_communicator) :: comm

  !$omp parallel shared(nthreads)
  nthreads = omp_get_num_threads()
  !$omp end parallel

  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world);

  ! make per-thread communicators
  !$omp parallel private(thrid, comm, np)

  ! make thread id 1-based
  thrid = omp_get_thread_num()+1
  print *, "thread ", thrid

  ! initialize shared datastructures
  !$omp master
  allocate(communicators(nthreads))  
  !$omp end master
  !$omp barrier

  ! allocate a communicator per thread and store in a shared array
  communicators(thrid) = ghex_comm_new()
  comm = communicators(thrid)

  ! do some work
  np = ghex_comm_progress(comm)

  ! cleanup per-thread
  call ghex_comm_delete(communicators(thrid))

  ! cleanup shared
  !$omp barrier
  !$omp master
  deallocate(communicators)
  !$omp end master

  !$omp end parallel

  call ghex_finalize()  
  call mpi_finalize(mpi_err)

END PROGRAM test_context
