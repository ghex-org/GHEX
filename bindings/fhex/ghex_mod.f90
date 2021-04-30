MODULE ghex_mod
  use iso_c_binding
  use ghex_defs
  use ghex_comm_mod
  use ghex_message_mod
  use ghex_structured_mod
  use ghex_cubed_sphere_mod

  implicit none

  interface
     subroutine ghex_init(nthreads, mpi_comm) bind(c)
       use iso_c_binding
       integer, value :: nthreads
       integer, value :: mpi_comm
     end subroutine ghex_init

     subroutine ghex_finalize() bind(c)
       use iso_c_binding
     end subroutine ghex_finalize
  end interface

END MODULE ghex_mod
