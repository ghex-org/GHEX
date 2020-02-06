MODULE ghex_structured_mod
  use iso_c_binding
  use ghex_comm_mod

  implicit none

  ! field descriptor - always associated with a domain
  type, bind(c) :: ghex_field_descriptor
     type(c_ptr)    :: data = c_null_ptr
     integer(c_int) ::  offset(3) = [-1]       ! by default - values from the halo
     integer(c_int) :: extents(3) = [-1]       ! by default - size of the local extents + halos
     integer(c_int) ::    halo(6) = [-1]       ! halo to be used for this field
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

  interface

     ! domain methods
     subroutine ghex_domain_add_field(domain_desc, field_desc) bind(c)
       use iso_c_binding
       import ghex_domain_descriptor, ghex_field_descriptor
       type(ghex_domain_descriptor) :: domain_desc
       type(ghex_field_descriptor) :: field_desc
     end subroutine ghex_domain_add_field

     ! those are CO functions, but right now they are also grid-specific
     type(ghex_communication_object) function ghex_struct_co_new() bind(c)
       import ghex_communication_object
     end function ghex_struct_co_new
     
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

    if (present(periodic)) then
       field_desc%periodic = periodic
    endif

  end subroutine ghex_field_init

END MODULE ghex_structured_mod
