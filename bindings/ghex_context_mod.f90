MODULE ghex_context_mod
  use iso_c_binding
  use ghex_comm_mod

  implicit none

  type, bind(c) :: ghex_context
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_context
  
  interface
     
     type(ghex_context) function context_new(nthreads, mpi_comm) bind(c)
       use iso_c_binding
       import ghex_context
       integer, value :: nthreads
       integer, value :: mpi_comm
     end function context_new

     subroutine context_delete(context) bind(c)
       use iso_c_binding
       import ghex_context
       ! reference, not a value - fortran variable is reset to null 
       type(ghex_context) :: context
     end subroutine context_delete

     integer(c_int) function context_rank(context) bind(c)
       use iso_c_binding
       import ghex_context
       type(ghex_context), value :: context
     end function context_rank

     integer(c_int) function context_size(context) bind(c)
       use iso_c_binding
       import ghex_context
       type(ghex_context), value :: context
     end function context_size
     
     type(ghex_communicator) function context_get_communicator(context) bind(c)
       use iso_c_binding
       use ghex_comm_mod
       import ghex_context
       type(ghex_context), value :: context
     end function context_get_communicator

  end interface

END MODULE ghex_context_mod
