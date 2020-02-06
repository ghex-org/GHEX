MODULE ghex_exchange_mod
  use iso_c_binding

  use ghex_structured_mod
  implicit none

  ! definition of the exchange, including physical fields and the communication pattern
  type, bind(c) :: ghex_exchange_descriptor
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_exchange_descriptor

  ! a handle to track a particular communication instance, supports wait()
  type, bind(c) :: ghex_exchange_handle
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_exchange_handle

  interface ghex_wait
     ! exchange handle 
     subroutine ghex_exchange_handle_wait(exchange_handle) bind(c)
       use iso_c_binding
       import ghex_exchange_handle
       type(ghex_exchange_handle), value :: exchange_handle
     end subroutine ghex_exchange_handle_wait
  end interface ghex_wait

  interface

     ! exchange descriptor methods
     type(ghex_exchange_descriptor) function ghex_exchange_desc_new_wrapped(domains_desc, n_domains) &
          bind(c, name="ghex_exchange_desc_new")
       use iso_c_binding
       import ghex_domain_descriptor, ghex_exchange_descriptor
       type(c_ptr), value :: domains_desc
       integer(c_int), value :: n_domains
     end function ghex_exchange_desc_new_wrapped

     type(ghex_exchange_handle) function ghex_exchange(co, exchange_desc) bind(c)
       import ghex_exchange_handle, ghex_communication_object, ghex_exchange_descriptor
       type(ghex_communication_object), value :: co
       type(ghex_exchange_descriptor), value :: exchange_desc
     end function ghex_exchange
     
  end interface

CONTAINS

  type(ghex_exchange_descriptor) function ghex_exchange_desc_new(domains_desc)
    type(ghex_domain_descriptor), dimension(:), target :: domains_desc
    
    ghex_exchange_desc_new = ghex_exchange_desc_new_wrapped(c_loc(domains_desc), size(domains_desc, 1));
  end function ghex_exchange_desc_new

END MODULE ghex_exchange_mod
