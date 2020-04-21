MODULE ghex_structured_mod
  use iso_c_binding
  use ghex_defs

  implicit none


  ! ---------------------
  ! --- module types
  ! ---------------------
  ! field descriptor - always associated with a domain
  type, bind(c) :: ghex_struct_field
     type(c_ptr)    :: data = c_null_ptr
     integer(c_int) ::  offset(3) = [-1]       ! by default - values from the halo
     integer(c_int) :: extents(3) = [-1]       ! by default - size of the local extents + halos
     integer(c_int) ::    halo(6) = [-1]       ! halo to be used for this field
     integer(c_int) :: periodic(3) = [0]
  end type ghex_struct_field

  ! computational domain: defines the iteration space, and the fields with their halos
  type, bind(c) :: ghex_struct_domain
     type(c_ptr)    :: fields = c_null_ptr  ! computational field data, opaque field not to be accessed by the user
     integer(c_int) :: id = -1
     integer(c_int) :: device_id = DeviceUnknown
     integer(c_int) :: first(3)             ! indices of the first LOCAL grid point, in global index space
     integer(c_int) :: last(3)              ! indices of the last LOCAL grid point, in global index space
     integer(c_int) :: gfirst(3) = [1,1,1]  ! indices of the first GLOBAL grid point, (1,1,1) by default
     integer(c_int) :: glast(3)             ! indices of the last GLOBAL grid point (model dimensions)
  end type ghex_struct_domain

  ! structured grid communication object
  type, bind(c) :: ghex_struct_communication_object
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_struct_communication_object

  ! definition of the exchange, including physical fields and the communication pattern
  type, bind(c) :: ghex_struct_exchange_descriptor
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_struct_exchange_descriptor

  ! a handle to track a particular communication instance, supports wait()
  type, bind(c) :: ghex_struct_exchange_handle
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_struct_exchange_handle


  ! ---------------------
  ! --- module C interfaces
  ! ---------------------
  interface
     type(ghex_struct_exchange_descriptor) function ghex_struct_exchange_desc_new_wrapped(domains_desc, n_domains) &
          bind(c, name="ghex_struct_exchange_desc_new")
       use iso_c_binding
       import ghex_struct_domain, ghex_struct_exchange_descriptor
       type(c_ptr), value :: domains_desc
       integer(c_int), value :: n_domains
     end function ghex_struct_exchange_desc_new_wrapped
  end interface


  ! ---------------------
  ! --- generic ghex interfaces
  ! ---------------------
  interface ghex_delete
     subroutine ghex_struct_exchange_desc_delete(exchange_desc) bind(c, name="ghex_obj_delete")
       use iso_c_binding
       import ghex_struct_exchange_descriptor
       type(ghex_struct_exchange_descriptor) :: exchange_desc
     end subroutine ghex_struct_exchange_desc_delete

     subroutine ghex_struct_exchange_handle_delete(exchange_handle) bind(c, name="ghex_obj_delete")
       use iso_c_binding
       import ghex_struct_exchange_handle
       type(ghex_struct_exchange_handle) :: exchange_handle
     end subroutine ghex_struct_exchange_handle_delete

     subroutine ghex_struct_domain_delete(domains_desc) bind(c)
       use iso_c_binding
       import ghex_struct_domain
       type(ghex_struct_domain) :: domains_desc
     end subroutine ghex_struct_domain_delete

     subroutine ghex_struct_co_delete(co) bind(c, name="ghex_obj_delete")
       use iso_c_binding
       import ghex_struct_communication_object
       ! reference, not a value - fortran variable is reset to null
       type(ghex_struct_communication_object) :: co
     end subroutine ghex_struct_co_delete
  end interface ghex_delete

  interface ghex_field_init
     procedure :: ghex_struct_field_init
  end interface ghex_field_init

  interface ghex_co_init
     subroutine ghex_struct_co_init(co) bind(c)
       import ghex_struct_communication_object
       type(ghex_struct_communication_object) :: co
     end subroutine ghex_struct_co_init
  end interface ghex_co_init

  interface ghex_domain_add_field
     subroutine ghex_struct_domain_add_field(domain_desc, field_desc) bind(c)
       use iso_c_binding
       import ghex_struct_domain, ghex_struct_field
       type(ghex_struct_domain) :: domain_desc
       type(ghex_struct_field)  :: field_desc
     end subroutine ghex_struct_domain_add_field
  end interface ghex_domain_add_field

  interface ghex_exchange_desc_new
     procedure :: ghex_struct_exchange_desc_new
  end interface ghex_exchange_desc_new

  interface ghex_exchange
     type(ghex_struct_exchange_handle) function ghex_struct_exchange(co, exchange_desc) bind(c)
       import ghex_struct_exchange_handle, ghex_struct_communication_object, ghex_struct_exchange_descriptor
       type(ghex_struct_communication_object), value :: co
       type(ghex_struct_exchange_descriptor), value :: exchange_desc
     end function ghex_struct_exchange
  end interface ghex_exchange

  interface ghex_wait
     subroutine ghex_struct_exchange_handle_wait(exchange_handle) bind(c)
       use iso_c_binding
       import ghex_struct_exchange_handle
       type(ghex_struct_exchange_handle) :: exchange_handle
     end subroutine ghex_struct_exchange_handle_wait
  end interface ghex_wait

CONTAINS

  ! generic interface, ghex_field_mod
  subroutine ghex_struct_field_init(field_desc, data, halo, offset, periodic)
    type(ghex_struct_field) :: field_desc
    real(ghex_fp_kind), dimension(:,:,:), target :: data
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

    if (present(periodic)) then
       field_desc%periodic = periodic
    endif

  end subroutine ghex_struct_field_init

  type(ghex_struct_exchange_descriptor) function ghex_struct_exchange_desc_new(domains_desc)
    type(ghex_struct_domain), dimension(:), target :: domains_desc
    ghex_struct_exchange_desc_new = ghex_struct_exchange_desc_new_wrapped(c_loc(domains_desc), size(domains_desc, 1));
  end function ghex_struct_exchange_desc_new

END MODULE ghex_structured_mod
