PROGRAM test_context

  use omp_lib
  use ghex_context_mod

  implicit none  
  
  include 'mpif.h'  

  integer :: mpi_err
  integer :: mpi_threading
  integer :: nthreads = 0
  type(ghex_context) :: context

  !$omp parallel shared(nthreads)
  nthreads = omp_get_num_threads()
  !$omp end parallel

  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)

  ! create a context object
  context = context_new(nthreads, mpi_comm_world)

  ! delete the ghex context
  call context_delete(context)

END PROGRAM test_context
