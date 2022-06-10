MODULE ghex_unstructured_mod

    !
    ! GridTools
    !
    ! Copyright (c) 2014-2021, ETH Zurich
    ! All rights reserved.
    !
    ! Please, refer to the LICENSE file in the root directory.
    ! SPDX-License-Identifier: BSD-3-Clause
    !

    use iso_c_binding
    use ghex_defs

    implicit none

    ! ---------------------
    ! --- module types
    ! ---------------------

    ! domain descriptor
    type, bind(c) :: ghex_unstruct_domain_desc
        integer(c_int) :: id = -1
        type(c_ptr) :: vertices = c_null_ptr
        integer(c_int) :: total_size = 0
        integer(c_int) :: inner_size = 0
        integer(c_int) :: levels = 1
    end type ghex_unstruct_domain_desc

    ! pattern
    type, bind(c) :: ghex_unstruct_pattern
        type(c_ptr) :: ptr = c_null_ptr
    end type ghex_unstruct_pattern

    ! ---------------------
    ! --- module C interfaces
    ! ---------------------

    interface
        subroutine ghex_unstruct_pattern_setup_impl(pattern, domain_descs, n_domains) bind(c)
            type(ghex_unstruct_pattern) :: pattern
            type(c_ptr) :: domain_descs
            integer(c_int) :: n_domains
        end subroutine ghex_unstruct_pattern_setup_impl
    end interface

    ! ---------------------
    ! --- generic ghex interfaces
    ! ---------------------

    interface ghex_unstruct_domain_desc_init
        procedure :: ghex_unstruct_domain_desc_init
    end interface ghex_unstruct_domain_desc_init

    interface ghex_unstruct_pattern_setup
        procedure :: ghex_unstruct_pattern_setup
    end interface ghex_unstruct_pattern_setup

CONTAINS

    subroutine ghex_unstruct_domain_desc_init(domain_desc, id, vertices, total_size, inner_size, levels)
        type(ghex_unstruct_domain_desc) :: domain_desc
        integer :: id
        integer, dimension(:), target :: vertices
        integer :: total_size
        integer :: inner_size
        integer, optional :: levels

        domain_desc%id = id
        domain_desc%vertices = c_loc(vertices)
        domain_desc%total_size = total_size
        domain_desc%inner_size = inner_size
        if (present(levels)) then
            domain_desc%levels = levels
        endif
    end subroutine ghex_unstruct_domain_desc_init

    subroutine ghex_unstruct_pattern_setup(pattern, domain_descs)
        type(ghex_unstruct_pattern) :: pattern
        type(ghex_unstruct_domain_desc), dimension(:), target :: domain_descs
        call ghex_unstruct_pattern_setup_impl(pattern, c_loc(domain_descs), size(domain_descs))
    end subroutine ghex_unstruct_pattern_setup

    ! --- HERE

    ! field descriptor
    type, bind(c) :: ghex_unstruct_field
        import ghex_unstruct_domain
        type(ghex_unstruct_domain) :: domain
        type(c_ptr) :: data = c_null_ptr
        integer(c_int) :: device_id = GhexDeviceUnknown
    end type ghex_unstruct_field

    ! unstructured grid communication object
    type, bind(c) :: ghex_unstruct_communication_object
        type(c_ptr) :: ptr = c_null_ptr
    end type ghex_unstruct_communication_object

    ! definition of the exchange, including physical fields and the communication pattern
    type, bind(c) :: ghex_unstruct_exchange_descriptor
        type(c_ptr) :: ptr = c_null_ptr
    end type ghex_unstruct_exchange_descriptor

    ! a handle to track a particular communication instance, supports wait()
    type, bind(c) :: ghex_unstruct_exchange_handle
        type(c_ptr) :: ptr = c_null_ptr
    end type ghex_unstruct_exchange_handle

    ! ---------------------
    ! --- generic ghex interfaces
    ! ---------------------

    interface ghex_free
        subroutine ghex_unstruct_exchange_desc_free(exchange_desc) bind(c, name="ghex_obj_free")
            use iso_c_binding
            import ghex_unstruct_exchange_descriptor
            type(ghex_unstruct_exchange_descriptor) :: exchange_desc
        end subroutine ghex_unstruct_exchange_desc_free

        subroutine ghex_unstruct_exchange_handle_free(exchange_handle) bind(c, name="ghex_obj_free")
            use iso_c_binding
            import ghex_unstruct_exchange_handle
            type(ghex_unstruct_exchange_handle) :: exchange_handle
        end subroutine ghex_unstruct_exchange_handle_free

        subroutine ghex_unstruct_domain_free(domains_desc) bind(c)
            use iso_c_binding
            import ghex_unstruct_domain
            type(ghex_unstruct_domain) :: domains_desc
        end subroutine ghex_unstruct_domain_free

        subroutine ghex_unstruct_co_free(co) bind(c, name="ghex_obj_free")
            use iso_c_binding
            import ghex_unstruct_communication_object
            ! reference, not a value - fortran variable is reset to null
            type(ghex_unstruct_communication_object) :: co
        end subroutine ghex_unstruct_co_free

        procedure :: ghex_unstruct_field_free
    end interface ghex_free

    interface ghex_field_init
        procedure :: ghex_unstruct_field_init
    end interface ghex_field_init

    interface ghex_co_init
        subroutine ghex_unstruct_co_init(co) bind(c)
            import ghex_unstruct_communication_object
            type(ghex_unstruct_communication_object) :: co
        end subroutine ghex_unstruct_co_init
    end interface ghex_co_init

    ! TO DO: TO BE CHANGED (should be ghex_exchange_desc_add_field)
    ! interface ghex_domain_add_field
        ! ...
    ! end interface ghex_domain_add_field

    interface ghex_exchange_desc_new
        procedure :: ghex_unstruct_exchange_desc_new
    end interface ghex_exchange_desc_new

    interface ghex_exchange
        type(ghex_unstruct_exchange_handle) function ghex_unstruct_exchange(co, exchange_desc) bind(c)
            import ghex_unstruct_exchange_handle, ghex_unstruct_communication_object, ghex_unstruct_exchange_descriptor
            type(ghex_unstruct_communication_object), value :: co
            type(ghex_unstruct_exchange_descriptor), value :: exchange_desc
        end function ghex_unstruct_exchange
    end interface ghex_exchange

    interface ghex_wait
        subroutine ghex_unstruct_exchange_handle_wait(exchange_handle) bind(c)
            use iso_c_binding
            import ghex_unstruct_exchange_handle
            type(ghex_unstruct_exchange_handle) :: exchange_handle
        end subroutine ghex_unstruct_exchange_handle_wait
    end interface ghex_wait

CONTAINS

    subroutine ghex_unstruct_field_init(field_desc)
        type(ghex_struct_field) :: field_desc
        ! TO DO
    end subroutine ghex_unstruct_field_init

    subroutine ghex_unstruct_field_free(field_desc)
        type(ghex_unstruct_field) :: field_desc
        ! TO DO
    end subroutine ghex_unstruct_field_free

    type(ghex_unstruct_exchange_descriptor) function ghex_unstruct_exchange_desc_new(domains_desc)
        type(ghex_unstruct_domain), target :: domains_desc
        ghex_unstruct_exchange_desc_new = ghex_unstruct_exchange_desc_new_wrapped(c_loc(domains_desc), 1);
    end function ghex_unstruct_exchange_desc_new

END MODULE ghex_unstructured_mod
