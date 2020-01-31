MODULE ghex_structured_mod
  use iso_c_binding

  use ghex_context_mod
  use ghex_comm_mod

  implicit none

  type, bind(c) :: ghex_pattern
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_pattern

  type, bind(c) :: ghex_domain_descriptor
     integer :: id
     integer :: first(3)   ! indices of the first LOCAL grid point, in global index space
     integer :: last(3)    ! indices of the last LOCAL grid point, in global index space
  end type ghex_domain_descriptor

  type, bind(c) :: ghex_field_descriptor
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_field_descriptor

  type, bind(c) :: ghex_communication_object
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_communication_object

  type, bind(c) :: ghex_exchange_handle
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_exchange_handle

  interface
     type(ghex_pattern) function ghex_make_pattern_wrapped(context, halo, domain_desc, ndomain_desc, periodic, &
          global_first, global_last) bind(c, name="ghex_make_pattern")
       use iso_c_binding
       import ghex_pattern, ghex_domain_descriptor, ghex_context
       type(ghex_context), value :: context
       integer(c_int), dimension(:) :: halo(6)
       type(c_ptr), value :: domain_desc
       integer(c_int), dimension(:) :: periodic(3)
       integer(c_int), value :: ndomain_desc
       integer(c_int), dimension(:) :: global_first(3), global_last(3)
     end function ghex_make_pattern_wrapped

     type(ghex_field_descriptor) function ghex_wrap_field_wrapped(domain_id, field, local_offset, field_extents) &
          bind(c, name="ghex_wrap_field")
       use iso_c_binding
       import ghex_field_descriptor
       integer(c_int), value :: domain_id
       type(c_ptr), value :: field
       integer(c_int), dimension(:) :: local_offset(3)
       integer(c_int), dimension(:) :: field_extents(3)
     end function ghex_wrap_field_wrapped

     type(ghex_communication_object) function ghex_make_communication_object(comm) bind(c)
       import ghex_communicator, ghex_communication_object
       type(ghex_communicator), value :: comm
     end function ghex_make_communication_object

     type(ghex_exchange_future) function ghex_exchange(co, pattern, field) bind(c)
       import ghex_exchange_future, ghex_communication_object, ghex_pattern, ghex_field_descriptor
       type(ghex_communication_object), value :: co
       type(ghex_pattern), value :: pattern
       type(ghex_field_descriptor), value :: field
     end function ghex_exchange

  end interface

CONTAINS

  type(ghex_pattern) function ghex_make_pattern(context, halo, domain_desc, periodic, global_first, global_last)
    type(ghex_context) :: context
    integer, dimension(:) :: halo(6)
    type(ghex_domain_descriptor), pointer, dimension(:) :: domain_desc
    integer, dimension(:) :: periodic(3)
    integer, dimension(:) :: global_first(3), global_last(3)

    ghex_make_pattern = ghex_make_pattern_wrapped(context, halo, c_loc(domain_desc), size(domain_desc, 1), periodic, &
         global_first, global_last)
  end function ghex_make_pattern
  
  type(ghex_field_descriptor) function ghex_wrap_field(domain_id, field, local_offset)
    integer :: domain_id
    real(8), dimension(:,:,:), pointer :: field
    integer, dimension(:) :: local_offset(3)

    ghex_wrap_field = ghex_wrap_field_wrapped(domain_id, c_loc(field), local_offset, shape(field, 4))
  end function ghex_wrap_field

END MODULE ghex_structured_mod
