MODULE ghex_structured_mod
  use iso_c_binding

  use ghex_comm_mod

  implicit none

  type, bind(c) :: ghex_pattern
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_pattern

  ! field descriptor - always associated with a domain
  type, bind(c) :: ghex_field_descriptor
     type(c_ptr)    :: data = c_null_ptr
     integer(c_int) :: offset(3) = [-1]        ! by default - values from the halo
     integer(c_int) :: extents(3) = [-1]       ! by default - size of the local extents + halos
     integer(c_int) :: halo(6) = [-1]          ! halo to be used for this field
     integer(c_int) :: periodic(3) = [0]
  end type ghex_field_descriptor

  integer, public, parameter :: DeviceUnknown = 0
  integer, public, parameter :: DeviceCPU = 1
  integer, public, parameter :: DeviceGPU = 2

  ! computational domain: defines the iteration space, and the fields with their halos
  type, bind(c) :: ghex_domain_descriptor
     integer(c_int) :: id = -1
     integer(c_int) :: device_id = DeviceUnknown
     integer(c_int) :: first(3)             ! indices of the first LOCAL grid point, in global index space
     integer(c_int) :: last(3)              ! indices of the last LOCAL grid point, in global index space
     integer(c_int) :: gfirst(3) = [1,1,1]  ! indices of the first GLOBAL grid point, (1,1,1) by default
     integer(c_int) :: glast(3)             ! indices of the last GLOBAL grid point (model dimensions)
     type(c_ptr)    :: fields = c_null_ptr  ! computational field data, opaque field not to be accessed by the user
  end type ghex_domain_descriptor

  type, bind(c) :: ghex_communication_object
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_communication_object

  type, bind(c) :: ghex_exchange_descriptor
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_exchange_descriptor

  type, bind(c) :: ghex_exchange_handle
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_exchange_handle

  interface

     subroutine ghex_domain_add_field(domain_desc, field_desc) bind(c)
       use iso_c_binding
       import ghex_domain_descriptor, ghex_field_descriptor
       type(ghex_domain_descriptor) :: domain_desc
       type(ghex_field_descriptor) :: field_desc
     end subroutine ghex_domain_add_field

     subroutine ghex_domain_delete(domains_desc) bind(c)
       use iso_c_binding
       import ghex_domain_descriptor, ghex_field_descriptor
       type(ghex_domain_descriptor) :: domains_desc
     end subroutine ghex_domain_delete

     type(ghex_exchange_descriptor) function ghex_exchange_new_wrapped(domains_desc, n_domains) bind(c, name="ghex_exchange_new")
       use iso_c_binding
       import ghex_domain_descriptor, ghex_exchange_descriptor
       type(c_ptr), value :: domains_desc
       integer(c_int), value :: n_domains
     end function ghex_exchange_new_wrapped

     subroutine ghex_exchange_delete(exchange_desc) bind(c)
       use iso_c_binding
       import ghex_exchange_descriptor
       type(ghex_exchange_descriptor) :: exchange_desc
     end subroutine ghex_exchange_delete

     type(ghex_pattern) function ghex_make_pattern_wrapped(halo, domain_desc, ndomain_desc, periodic, &
          global_first, global_last) bind(c, name="ghex_make_pattern")
       use iso_c_binding
       import ghex_pattern, ghex_domain_descriptor
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

     ! those are CO functions, but right now they are also grid-specific
     type(ghex_communication_object) function ghex_struct_co_new() bind(c)
       import ghex_communication_object
     end function ghex_struct_co_new

     subroutine ghex_struct_co_delete(co) bind(c)
       use iso_c_binding
       import ghex_communication_object
       ! reference, not a value - fortran variable is reset to null
       type(ghex_communication_object) :: co
     end subroutine ghex_struct_co_delete
     
     type(ghex_exchange_handle) function ghex_struct_exchange(co, pattern, field) bind(c)
       import ghex_exchange_handle, ghex_communication_object, ghex_pattern, ghex_field_descriptor
       type(ghex_communication_object), value :: co
       type(ghex_pattern), value :: pattern
       type(ghex_field_descriptor), value :: field
     end function ghex_struct_exchange
     
  end interface

CONTAINS

  subroutine ghex_field_init(field_desc, data, halo, offset, periodic)
    type(ghex_field_descriptor) :: field_desc
    real(8), dimension(:,:,:), target :: data
    integer :: halo(6)
    integer, optional :: offset(3)
    integer, optional :: periodic(3)

    field_desc%data = c_loc(data)
    field_desc%halo = halo
    field_desc%extents = shape(data, 4)
    
    if (present(offset)) then
       field_desc%offset = offset
    else
       field_desc%offset = [halo(1), halo(3), halo(5)]
    endif

    if (present(offset)) then
       field_desc%offset = offset
    endif

  end subroutine ghex_field_init

  type(ghex_exchange_descriptor) function ghex_exchange_new(domains_desc)
    type(ghex_domain_descriptor), dimension(:), target :: domains_desc
    
    ghex_exchange_new = ghex_exchange_new_wrapped(c_loc(domains_desc), size(domains_desc, 1));
  end function ghex_exchange_new

  type(ghex_pattern) function ghex_make_pattern(halo, domain_desc, periodic, global_first, global_last)
    integer, dimension(:) :: halo(6)
    type(ghex_domain_descriptor), pointer, dimension(:) :: domain_desc
    integer, dimension(:) :: periodic(3)
    integer, dimension(:) :: global_first(3), global_last(3)

    ghex_make_pattern = ghex_make_pattern_wrapped(halo, c_loc(domain_desc), size(domain_desc, 1), periodic, &
         global_first, global_last)
  end function ghex_make_pattern
  
  type(ghex_field_descriptor) function ghex_wrap_field(domain_id, field, local_offset)
    integer :: domain_id
    real(8), dimension(:,:,:), pointer :: field
    integer, dimension(:) :: local_offset(3)

    ghex_wrap_field = ghex_wrap_field_wrapped(domain_id, c_loc(field), local_offset, shape(field, 4))
  end function ghex_wrap_field

END MODULE ghex_structured_mod
