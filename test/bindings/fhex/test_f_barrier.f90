!
! ghex-org
!
! Copyright (c) 2014-2023, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause
!
PROGRAM test_f_barrier
  use iso_fortran_env
  use omp_lib
  use ghex_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err
  integer :: mpi_threading, mpi_size, mpi_rank
  integer :: nthreads

  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)
  if (MPI_THREAD_MULTIPLE /= mpi_threading) then
    stop "MPI does not support multithreading"
  end if
  call mpi_comm_size (mpi_comm_world, mpi_size, mpi_err)
  call mpi_comm_rank (mpi_comm_world, mpi_rank, mpi_err)

  !$omp parallel shared(nthreads)
  nthreads = omp_get_num_threads()
  !$omp end parallel

  if (mpi_rank==0) then
    print *, mpi_size, "ranks and", nthreads, "threads per rank"
  end if

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world);

  !$omp parallel

  ! call MT barrier
  call ghex_barrier(GhexBarrierGlobal)

  !$omp end parallel

  if (mpi_rank==0) then
    print *, "ghex_comm_barrier DONE"
  end if

  call ghex_finalize()  
  call mpi_finalize(mpi_err)

END PROGRAM test_f_barrier
