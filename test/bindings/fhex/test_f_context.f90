!
! ghex-org
!
! Copyright (c) 2014-2023, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause
!
PROGRAM test_context
  use ghex_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err
  integer :: mpi_threading
  integer :: nthreads

  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)

  ! init ghex
  nthreads = 1
  call ghex_init(nthreads, mpi_comm_world)

  call ghex_finalize()
  call mpi_finalize(mpi_err)

END PROGRAM test_context
