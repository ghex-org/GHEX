!
! ghex-org
!
! Copyright (c) 2014-2023, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause
!
MODULE ghex_mod
  use iso_c_binding
  use ghex_defs
  !use ghex_comm_mod
  !use ghex_message_mod
  !use ghex_structured_mod
  !use ghex_cubed_sphere_mod

  implicit none

  interface
     subroutine ghex_init(nthreads, mpi_comm) bind(c)
       use iso_c_binding
       integer, value :: nthreads, mpi_comm
     end subroutine ghex_init

     subroutine ghex_finalize() bind(c)
       use iso_c_binding
     end subroutine ghex_finalize

#ifdef GHEX_ENABLE_BARRIER
     subroutine ghex_barrier(type) bind(c)
       use iso_c_binding
       integer, value :: type
     end subroutine ghex_barrier
#endif
  end interface

END MODULE ghex_mod
