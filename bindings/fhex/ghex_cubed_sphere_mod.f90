!
! ghex-org
!
! Copyright (c) 2014-2023, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause
!
MODULE ghex_cubed_sphere_mod
  use iso_c_binding
  use ghex_defs
  use ghex_comm_mod

  implicit none


  ! ---------------------
  ! --- module types
  ! ---------------------
  ! field descriptor - always associated with a domain
  type, bind(c) :: ghex_cubed_sphere_field
     type(c_ptr)    :: data = c_null_ptr
     integer(c_int) ::  offset(3) = -1         ! by default - values from the halo
     integer(c_int) :: extents(3) = -1         ! by default - size of the local extents + halos
     integer(c_int) ::    halo(4) = -1         ! halo to be used for this field
     integer(c_int) :: n_components = 1        ! number of field components
     integer(c_int) ::     layout = GhexLayoutFieldLast
     logical(c_bool) :: is_vector = .false.    ! is this a vector field
  end type ghex_cubed_sphere_field

  ! computational domain: defines the iteration space, and the fields with their halos
  type, bind(c) :: ghex_cubed_sphere_domain
     type(c_ptr)    :: fields = c_null_ptr  ! computational field data, opaque field not to be accessed by the user
     integer(c_int) :: tile = -1
     integer(c_int) :: device_id = GhexDeviceUnknown
     integer(c_int) :: cube(2)              ! local grid dimensions
     integer(c_int) :: first(2)             ! indices of the first LOCAL grid point, in global index space
     integer(c_int) :: last(2)              ! indices of the last LOCAL grid point, in global index space
  end type ghex_cubed_sphere_domain

  type, bind(c) :: ghex_cubed_sphere_communication_object
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_cubed_sphere_communication_object

  ! definition of the exchange, including physical fields and the communication pattern
  type, bind(c) :: ghex_cubed_sphere_exchange_descriptor
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_cubed_sphere_exchange_descriptor

  ! a handle to track a particular communication instance, supports wait()
  type, bind(c) :: ghex_cubed_sphere_exchange_handle
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_cubed_sphere_exchange_handle


  ! ---------------------
  ! --- module C interfaces
  ! ---------------------
  interface
     type(ghex_cubed_sphere_exchange_descriptor) function ghex_cubed_sphere_exchange_desc_new_wrapped(domains_desc, n_domains) &
          bind(c, name="ghex_cubed_sphere_exchange_desc_new")
       use iso_c_binding
       import ghex_cubed_sphere_domain, ghex_cubed_sphere_exchange_descriptor
       type(c_ptr), value :: domains_desc
       integer(c_int), value :: n_domains
     end function ghex_cubed_sphere_exchange_desc_new_wrapped
  end interface


  ! ---------------------
  ! --- generic ghex interfaces
  ! ---------------------
  interface ghex_free
     subroutine ghex_cubed_sphere_exchange_desc_free(exchange_desc) bind(c, name="ghex_obj_free")
       use iso_c_binding
       import ghex_cubed_sphere_exchange_descriptor
       type(ghex_cubed_sphere_exchange_descriptor) :: exchange_desc
     end subroutine ghex_cubed_sphere_exchange_desc_free

     subroutine ghex_cubed_sphere_exchange_handle_free(exchange_handle) bind(c, name="ghex_obj_free")
       use iso_c_binding
       import ghex_cubed_sphere_exchange_handle
       type(ghex_cubed_sphere_exchange_handle) :: exchange_handle
     end subroutine ghex_cubed_sphere_exchange_handle_free

     subroutine ghex_cubed_sphere_domain_free(domains_desc) bind(c)
       use iso_c_binding
       import ghex_cubed_sphere_domain
       type(ghex_cubed_sphere_domain) :: domains_desc
     end subroutine ghex_cubed_sphere_domain_free

     subroutine ghex_cubed_sphere_co_free(co) bind(c, name="ghex_obj_free")
       use iso_c_binding
       import ghex_cubed_sphere_communication_object
       ! reference, not a value - fortran variable is reset to null
       type(ghex_cubed_sphere_communication_object) :: co
     end subroutine ghex_cubed_sphere_co_free

     procedure :: ghex_cubed_sphere_field_free
  end interface ghex_free

  interface ghex_domain_init
     procedure :: ghex_cubed_sphere_domain_init
  end interface ghex_domain_init

  interface ghex_field_init
     procedure :: ghex_cubed_sphere_field_init
     procedure :: ghex_cubed_sphere_field_init_comp
  end interface ghex_field_init

  interface ghex_co_init
     subroutine ghex_cubed_sphere_co_init(co, comm) bind(c)
       import ghex_cubed_sphere_communication_object, ghex_communicator
       type(ghex_cubed_sphere_communication_object) :: co
       type(ghex_communicator), value :: comm
     end subroutine ghex_cubed_sphere_co_init
  end interface ghex_co_init

  interface ghex_domain_add_field
     subroutine ghex_cubed_sphere_domain_add_field(domain_desc, field_desc) bind(c)
       use iso_c_binding
       import ghex_cubed_sphere_domain, ghex_cubed_sphere_field
       type(ghex_cubed_sphere_domain) :: domain_desc
       type(ghex_cubed_sphere_field)  :: field_desc
     end subroutine ghex_cubed_sphere_domain_add_field
  end interface ghex_domain_add_field

  interface ghex_exchange_desc_new
     procedure :: ghex_cubed_sphere_exchange_desc_new
     procedure :: ghex_cubed_sphere_exchange_desc_array_new
  end interface ghex_exchange_desc_new

  interface ghex_exchange
     type(ghex_cubed_sphere_exchange_handle) function ghex_cubed_sphere_exchange(co, exchange_desc) bind(c)
       import ghex_cubed_sphere_exchange_handle, ghex_cubed_sphere_communication_object, ghex_cubed_sphere_exchange_descriptor
       type(ghex_cubed_sphere_communication_object), value :: co
       type(ghex_cubed_sphere_exchange_descriptor), value :: exchange_desc
     end function ghex_cubed_sphere_exchange
  end interface ghex_exchange

  interface ghex_wait
     subroutine ghex_cubed_sphere_exchange_handle_wait(exchange_handle) bind(c)
       use iso_c_binding
       import ghex_cubed_sphere_exchange_handle
       type(ghex_cubed_sphere_exchange_handle) :: exchange_handle
     end subroutine ghex_cubed_sphere_exchange_handle_wait
  end interface ghex_wait

CONTAINS

  subroutine ghex_cubed_sphere_domain_init(domain_desc, tile, dims, first, last, device_id)
    type(ghex_cubed_sphere_domain) :: domain_desc
    integer :: tile, dims(2), first(2), last(2)
    integer, optional :: device_id

    if (present(device_id)) then
      domain_desc%device_id = device_id
    else
      domain_desc%device_id = GhexDeviceCPU
    end if

    domain_desc%tile  = tile
    domain_desc%cube  = dims
    domain_desc%first = first
    domain_desc%last  = last
  end subroutine ghex_cubed_sphere_domain_init

  subroutine ghex_cubed_sphere_field_init(field_desc, data, halo, offset)
    type(ghex_cubed_sphere_field) :: field_desc
    real(ghex_fp_kind), dimension(:,:,:), target :: data
    integer :: halo(4)
    integer, optional :: offset(3)

    field_desc%data = c_loc(data)
    field_desc%halo = halo
    field_desc%extents = shape(data, 4)
    field_desc%n_components = 1
    field_desc%is_vector = .false.
    field_desc%layout = GhexLayoutFieldLast

    if (present(offset)) then
       field_desc%offset = offset
    else
       field_desc%offset = [halo(1), halo(3), 0]
    endif
  end subroutine ghex_cubed_sphere_field_init

  subroutine ghex_cubed_sphere_field_init_comp(field_desc, data, halo, offset, layout, is_vector)
    type(ghex_cubed_sphere_field) :: field_desc
    real(ghex_fp_kind), dimension(:,:,:,:), target :: data
    integer :: halo(4)
    integer, optional :: offset(3)
    integer, optional :: layout
    logical, optional :: is_vector
    integer(4) :: extents(4)

    field_desc%data = c_loc(data)
    field_desc%halo = halo

    if (present(offset)) then
       field_desc%offset = offset
    else
       field_desc%offset = [halo(1), halo(3), 0]
    endif

    if (present(layout)) then
       field_desc%layout = layout
    else
       field_desc%layout = GhexLayoutFieldLast
    endif

    extents = shape(data, 4)
    if (field_desc%layout == GhexLayoutFieldLast) then
      field_desc%extents = extents(1:3)
      field_desc%n_components = size(data, 4)
    else
      field_desc%extents = extents(2:4)
      field_desc%n_components = size(data, 1)
    end if
    
    if (present(is_vector)) then
       field_desc%is_vector = is_vector
    else
       field_desc%is_vector = .false.
    endif
  end subroutine ghex_cubed_sphere_field_init_comp

  subroutine ghex_cubed_sphere_field_free(field_desc)
    type(ghex_cubed_sphere_field) :: field_desc
    field_desc%data         = c_null_ptr
    field_desc%offset(:)    = -1
    field_desc%extents(:)   = -1
    field_desc%halo(:)      = -1
    field_desc%n_components = 1
    field_desc%layout       = GhexLayoutFieldLast
    field_desc%is_vector    = .false.
  end subroutine ghex_cubed_sphere_field_free

  type(ghex_cubed_sphere_exchange_descriptor) function ghex_cubed_sphere_exchange_desc_array_new(domains_desc)
    type(ghex_cubed_sphere_domain), dimension(:), target :: domains_desc
    ghex_cubed_sphere_exchange_desc_array_new = ghex_cubed_sphere_exchange_desc_new_wrapped(c_loc(domains_desc), size(domains_desc, 1));
  end function ghex_cubed_sphere_exchange_desc_array_new

  type(ghex_cubed_sphere_exchange_descriptor) function ghex_cubed_sphere_exchange_desc_new(domains_desc)
    type(ghex_cubed_sphere_domain), target :: domains_desc
    ghex_cubed_sphere_exchange_desc_new = ghex_cubed_sphere_exchange_desc_new_wrapped(c_loc(domains_desc), 1);
  end function ghex_cubed_sphere_exchange_desc_new

END MODULE ghex_cubed_sphere_mod
