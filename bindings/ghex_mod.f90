MODULE ghex_mod
  use iso_c_binding
  use ghex_defs
  use ghex_comm_mod
  use ghex_message_mod
  use ghex_exchange_mod

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

  interface ghex_initialized
     procedure :: ghex_initialized_exchange_desc
     procedure :: ghex_initialized_communication_object     
  end interface ghex_initialized

  interface ghex_delete

     subroutine ghex_comm_delete(comm) bind(c, name="ghex_obj_delete")
       use iso_c_binding
       import ghex_communicator
       type(ghex_communicator) :: comm
     end subroutine ghex_comm_delete

     subroutine ghex_exchange_desc_delete(exchange_desc) bind(c, name="ghex_obj_delete")
       use iso_c_binding
       import ghex_exchange_descriptor
       type(ghex_exchange_descriptor) :: exchange_desc
     end subroutine ghex_exchange_desc_delete

     subroutine ghex_exchange_handle_delete(exchange_handle) bind(c, name="ghex_obj_delete")
       use iso_c_binding
       import ghex_exchange_handle
       type(ghex_exchange_handle) :: exchange_handle
     end subroutine ghex_exchange_handle_delete

     subroutine ghex_domain_delete(domains_desc) bind(c)
       use iso_c_binding
       import ghex_domain_descriptor, ghex_field_descriptor
       type(ghex_domain_descriptor) :: domains_desc
     end subroutine ghex_domain_delete

     subroutine ghex_struct_co_delete(co) bind(c, name="ghex_obj_delete")
       use iso_c_binding
       import ghex_communication_object
       ! reference, not a value - fortran variable is reset to null
       type(ghex_communication_object) :: co
     end subroutine ghex_struct_co_delete

     subroutine ghex_message_delete(message) bind(c)
       use iso_c_binding
       import ghex_message
       ! reference, not a value - fortran variable is reset to null 
       type(ghex_message) :: message
     end subroutine ghex_message_delete

  end interface ghex_delete

END MODULE ghex_mod
