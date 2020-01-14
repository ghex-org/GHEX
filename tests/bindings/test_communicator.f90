PROGRAM test_context

  use omp_lib
  use ghex_context_mod
  use ghex_comm_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err
  integer :: mpi_threading
  integer :: nthreads = 0, thrid
  integer :: np = 0
  type(ghex_context) :: context
  type(ghex_communicator), dimension(:), pointer :: comm

  !$omp parallel shared(nthreads)
  nthreads = omp_get_num_threads()
  !$omp end parallel

  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)

  ! create a context object
  context = context_new(nthreads, mpi_comm_world);

  ! make a communicator - per-thread
  !$omp parallel private(thrid, comm, np)

  !$omp master
  allocate(comm(nthreads))
  !$omp end master

  !$omp barrier
  thrid = omp_get_thread_num()
  ! comm(thrid) => context_get_communicator(context)
  ! np = comm_progress(comm)

  ! doesn't really have to explicitly destruct comm:
  ! it is done automatically, when comm is out of scope
  ! call comm_delete(comm)
  !$omp end parallel

  ! delete the ghex context
  call context_delete(context)

END PROGRAM test_context
