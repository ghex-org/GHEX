PROGRAM test_context
  use omp_lib
  use ghex_mod
  use ghex_comm_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err
  integer :: mpi_threading
  integer :: nthreads = 0
  type(ghex_progress_status) :: ps
  type(ghex_communicator) :: comm

  !$omp parallel shared(nthreads)
  nthreads = omp_get_num_threads()
  !$omp end parallel

  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world);

  ! make per-thread communicators
  !$omp parallel private(comm, ps)

  ! allocate a communicator per thread and store it
  comm = ghex_comm_new()

  ! do some work
  ps = ghex_comm_progress(comm)

  ! cleanup per-thread
  call ghex_free(comm)

  !$omp end parallel

  call ghex_finalize()  
  call mpi_finalize(mpi_err)

END PROGRAM test_context
