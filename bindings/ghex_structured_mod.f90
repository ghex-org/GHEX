MODULE ghex_structured_mod
  use iso_c_binding
  use ghex_defs
  use ghex_comm_mod

  implicit none


  ! ---------------------
  ! --- module types
  ! ---------------------
  ! field descriptor - always associated with a domain
  type, bind(c) :: ghex_struct_field
     type(c_ptr)    :: data = c_null_ptr
     integer(c_int) ::  offset(3) = -1         ! by default - values from the halo
     integer(c_int) :: extents(3) = -1         ! by default - size of the local extents + halos
     integer(c_int) ::    halo(6) = -1         ! halo to be used for this field
     integer(c_int) :: periodic(3) = .false.
     integer(c_int) :: n_components = 1        ! number of field components
     integer(c_int) ::     layout = LayoutFieldLast
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

     ! optional, used by the staged communication object
     integer(c_int) :: cart_comm = 0        ! cartesian communicator
     integer(c_int) :: cart_order = 0       ! cartesian communicator
     integer(c_int) :: cart_dim(3) = [0,0,0]! rank dimensions in the cartesian communicator
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
  interface ghex_free
     subroutine ghex_struct_exchange_desc_free(exchange_desc) bind(c, name="ghex_obj_free")
       use iso_c_binding
       import ghex_struct_exchange_descriptor
       type(ghex_struct_exchange_descriptor) :: exchange_desc
     end subroutine ghex_struct_exchange_desc_free

     subroutine ghex_struct_exchange_handle_free(exchange_handle) bind(c, name="ghex_obj_free")
       use iso_c_binding
       import ghex_struct_exchange_handle
       type(ghex_struct_exchange_handle) :: exchange_handle
     end subroutine ghex_struct_exchange_handle_free

     subroutine ghex_struct_domain_free(domains_desc) bind(c)
       use iso_c_binding
       import ghex_struct_domain
       type(ghex_struct_domain) :: domains_desc
     end subroutine ghex_struct_domain_free

     subroutine ghex_struct_co_free(co) bind(c, name="ghex_obj_free")
       use iso_c_binding
       import ghex_struct_communication_object
       ! reference, not a value - fortran variable is reset to null
       type(ghex_struct_communication_object) :: co
     end subroutine ghex_struct_co_free

     procedure :: ghex_struct_field_free
  end interface ghex_free

  interface ghex_domain_init
     procedure :: ghex_struct_domain_init
  end interface ghex_domain_init

  interface ghex_field_init
     procedure :: ghex_struct_field_init
     procedure :: ghex_struct_field_init_comp
  end interface ghex_field_init

  interface ghex_co_init
     subroutine ghex_struct_co_init(co, comm) bind(c)
       import ghex_struct_communication_object, ghex_communicator
       type(ghex_struct_communication_object) :: co
       type(ghex_communicator), value :: comm
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
     procedure :: ghex_struct_exchange_desc_array_new
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

  subroutine ghex_struct_domain_init(domain_desc, id, first, last, gfirst, glast, cart_comm, cart_order, cart_dim, device_id)
    type(ghex_struct_domain) :: domain_desc
    integer :: id
    integer :: first(3)
    integer :: last(3)
    integer :: gfirst(3)
    integer :: glast(3)
    integer, optional :: cart_comm
    integer, optional :: cart_order
    integer, optional :: cart_dim(3)
    integer, optional :: device_id

    if (present(cart_comm).or.present(cart_dim).or.present(cart_order)) then
      if (.not.(present(cart_dim).and.present(cart_comm).and.present(cart_order))) then
        write (*,*) "ERROR: cart_comm, cart_order, and cart_dim arguments must ALL be present"
        call exit(1)
      end if
      domain_desc%cart_dim = cart_dim
      domain_desc%cart_comm = cart_comm
      domain_desc%cart_order = cart_order
    end if
    
    if (present(device_id)) then
      domain_desc%device_id = device_id
    else
      domain_desc%device_id = DeviceCPU
    end if

    domain_desc%id = id
    domain_desc%first = first
    domain_desc%last = last
    domain_desc%gfirst = gfirst
    domain_desc%glast = glast
  end subroutine ghex_struct_domain_init

  subroutine ghex_struct_field_init(field_desc, data, halo, offset, periodic)
    type(ghex_struct_field) :: field_desc
    real(ghex_fp_kind), dimension(:,:,:), target :: data
    integer :: halo(6)
    integer, optional :: offset(3)
    logical, optional :: periodic(3)

    field_desc%data = c_loc(data)
    field_desc%halo = halo
    field_desc%extents = shape(data, 4)
    field_desc%n_components = 1
    field_desc%layout = LayoutFieldLast

    if (present(offset)) then
      field_desc%offset = offset
    else
      field_desc%offset = [halo(1), halo(3), halo(5)]
    endif

    if (present(periodic)) then
      field_desc%periodic = periodic
    endif
  end subroutine ghex_struct_field_init

  subroutine ghex_struct_field_init_comp(field_desc, data, halo, offset, periodic, layout)
    type(ghex_struct_field) :: field_desc
    real(ghex_fp_kind), dimension(:,:,:,:), target :: data
    integer :: halo(6)
    integer, optional :: offset(3)
    logical, optional :: periodic(3)
    integer, optional :: layout
    integer(4) :: extents(4)

    field_desc%data = c_loc(data)
    field_desc%halo = halo

    if (present(offset)) then
      field_desc%offset = offset
    else
      field_desc%offset = [halo(1), halo(3), halo(5)]
    endif

    if (present(periodic)) then
      field_desc%periodic = periodic
    endif

    if (present(layout)) then
      field_desc%layout = layout
    else
      field_desc%layout = LayoutFieldLast
    endif

    extents = shape(data, 4)
    if (field_desc%layout == LayoutFieldLast) then
      field_desc%extents = extents(1:3)
      field_desc%n_components = size(data, 4)
    else
      field_desc%extents = extents(2:4)
      field_desc%n_components = size(data, 1)
    end if
  end subroutine ghex_struct_field_init_comp

  subroutine ghex_struct_field_free(field_desc)
    type(ghex_struct_field) :: field_desc
    field_desc%data         = c_null_ptr
    field_desc%offset(:)    = -1
    field_desc%extents(:)   = -1
    field_desc%halo(:)      = -1
    field_desc%periodic(:)  = .false.
    field_desc%layout       = LayoutFieldLast
  end subroutine ghex_struct_field_free

  type(ghex_struct_exchange_descriptor) function ghex_struct_exchange_desc_array_new(domains_desc)
    type(ghex_struct_domain), dimension(:), target :: domains_desc
    ghex_struct_exchange_desc_array_new = ghex_struct_exchange_desc_new_wrapped(c_loc(domains_desc), size(domains_desc, 1));
  end function ghex_struct_exchange_desc_array_new

  type(ghex_struct_exchange_descriptor) function ghex_struct_exchange_desc_new(domains_desc)
    type(ghex_struct_domain), target :: domains_desc
    ghex_struct_exchange_desc_new = ghex_struct_exchange_desc_new_wrapped(c_loc(domains_desc), 1);
  end function ghex_struct_exchange_desc_new

END MODULE ghex_structured_mod
