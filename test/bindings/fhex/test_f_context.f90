PROGRAM test_context
    use ghex_mod

    implicit none  

    include 'mpif.h'  

    integer :: mpi_err
    integer :: mpi_threading
    integer :: nthreads = 0
 
    call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)

    ! init ghex
    call ghex_init(mpi_comm_world)

    call ghex_finalize()
    call mpi_finalize(mpi_err)

END PROGRAM test_context
