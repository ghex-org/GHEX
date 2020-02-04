MODULE ghex_mod
  use iso_c_binding
  use ghex_comm_mod

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

     type(ghex_communicator) function ghex_get_communicator() bind(c)
       use iso_c_binding
       use ghex_comm_mod
     end function ghex_get_communicator

  end interface

END MODULE ghex_mod
