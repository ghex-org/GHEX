!
! ghex-org
!
! Copyright (c) 2014-2023, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause
!
MODULE ghex_unstructured_mod

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
        type(c_ptr) :: outer_indices = c_null_ptr
        integer(c_int) :: outer_size = 0
        integer(c_int) :: levels = 1
    end type ghex_unstruct_domain_desc

    ! pattern
    type, bind(c) :: ghex_unstruct_pattern
        type(c_ptr) :: ptr = c_null_ptr
    end type ghex_unstruct_pattern

    ! field descriptor
    type, bind(c) :: ghex_unstruct_field_desc
        integer(c_int) :: domain_id = -1
        integer(c_int) :: domain_size = 0
        integer(c_int) :: levels = 1
        type(c_ptr) :: field = c_null_ptr
    end type ghex_unstruct_field_desc

    ! communication object
    type, bind(c) :: ghex_unstruct_communication_object
        type(c_ptr) :: ptr = c_null_ptr
    end type ghex_unstruct_communication_object

    ! exchange args
    type, bind(c) :: ghex_unstruct_exchange_args
        type(c_ptr) :: ptr = c_null_ptr
    end type ghex_unstruct_exchange_args

    ! exchange handle
    type, bind(c) :: ghex_unstruct_exchange_handle
        type(c_ptr) :: ptr = c_null_ptr
    end type ghex_unstruct_exchange_handle

    ! ---------------------
    ! --- generic ghex interfaces
    ! ---------------------

    interface ghex_unstruct_domain_desc_init
        procedure :: ghex_unstruct_domain_desc_init
    end interface ghex_unstruct_domain_desc_init

    interface ghex_unstruct_field_desc_init
        procedure :: ghex_unstruct_field_desc_init
    end interface ghex_unstruct_field_desc_init

    interface ghex_unstruct_pattern_setup
        procedure :: ghex_unstruct_pattern_setup
    end interface ghex_unstruct_pattern_setup

    interface ghex_free
        subroutine ghex_unstruct_pattern_free(pattern) bind(c, name="ghex_obj_free")
            import ghex_unstruct_pattern
            type(ghex_unstruct_pattern) :: pattern
        end subroutine ghex_unstruct_pattern_free

        subroutine ghex_unstruct_communication_object_free(co) bind(c, name="ghex_obj_free")
            import ghex_unstruct_communication_object
            type(ghex_unstruct_communication_object) :: co
        end subroutine ghex_unstruct_communication_object_free

        subroutine ghex_unstruct_exchange_args_free(args) bind(c, name="ghex_obj_free")
            import ghex_unstruct_exchange_args
            type(ghex_unstruct_exchange_args) :: args
        end subroutine ghex_unstruct_exchange_args_free

        subroutine ghex_unstruct_exchange_handle_free(h) bind(c, name="ghex_obj_free")
            import ghex_unstruct_exchange_handle
            type(ghex_unstruct_exchange_handle) :: h
        end subroutine ghex_unstruct_exchange_handle_free
    end interface ghex_free

    interface ghex_clear
        procedure :: ghex_unstruct_domain_desc_clear
        procedure :: ghex_unstruct_field_desc_clear
    end interface ghex_clear

    ! ---------------------
    ! --- module C interfaces
    ! ---------------------

    interface
        subroutine ghex_unstruct_pattern_setup_impl(pattern, domain_descs, n_domains) bind(c)
            use iso_c_binding, only: c_ptr, c_int
            import ghex_unstruct_pattern
            type(ghex_unstruct_pattern) :: pattern
            type(c_ptr), value :: domain_descs ! TO DO: value?
            integer(c_int), value :: n_domains ! TO DO: value?
        end subroutine ghex_unstruct_pattern_setup_impl
    end interface

    interface
        subroutine ghex_unstruct_communication_object_init(co) bind(c)
            import ghex_unstruct_communication_object
            type(ghex_unstruct_communication_object) :: co
        end subroutine ghex_unstruct_communication_object_init
    end interface

    interface
        subroutine ghex_unstruct_exchange_args_init(args) bind(c)
            import ghex_unstruct_exchange_args
            type(ghex_unstruct_exchange_args) :: args
        end subroutine ghex_unstruct_exchange_args_init
    end interface

    interface
        subroutine ghex_unstruct_exchange_args_add(args, pattern, field_desc) bind(c)
            import ghex_unstruct_exchange_args, ghex_unstruct_pattern, ghex_unstruct_field_desc
            type(ghex_unstruct_exchange_args) :: args
            type(ghex_unstruct_pattern) :: pattern
            type(ghex_unstruct_field_desc) :: field_desc
        end subroutine ghex_unstruct_exchange_args_add
    end interface

    interface
        type(ghex_unstruct_exchange_handle) function ghex_unstruct_exchange(co, args) bind(c)
            import ghex_unstruct_exchange_handle, ghex_unstruct_communication_object, ghex_unstruct_exchange_args
            type(ghex_unstruct_communication_object) :: co
            type(ghex_unstruct_exchange_args) :: args
        end function ghex_unstruct_exchange
    end interface

    interface
        subroutine ghex_unstruct_exchange_handle_wait(h) bind(c)
            import ghex_unstruct_exchange_handle
            type(ghex_unstruct_exchange_handle) :: h
        end subroutine ghex_unstruct_exchange_handle_wait
    end interface

CONTAINS

    subroutine ghex_unstruct_domain_desc_init(domain_desc, id, vertices, total_size, outer_indices, outer_size, levels)
        type(ghex_unstruct_domain_desc) :: domain_desc
        integer :: id
        integer, dimension(:), target :: vertices
        integer :: total_size
        integer, dimension(:), target :: outer_indices
        integer :: outer_size
        integer, optional :: levels

        domain_desc%id = id
        domain_desc%vertices = c_loc(vertices)
        domain_desc%total_size = total_size
        domain_desc%outer_indices = c_loc(outer_indices)
        domain_desc%outer_size = outer_size
        if (present(levels)) then
            domain_desc%levels = levels
        endif
    end subroutine ghex_unstruct_domain_desc_init

    subroutine ghex_unstruct_field_desc_init(field_desc, domain_desc, field)
        type(ghex_unstruct_field_desc) :: field_desc
        type(ghex_unstruct_domain_desc) :: domain_desc
        real(ghex_fp_kind), dimension(:,:), target :: field

        field_desc%domain_id = domain_desc%id
        field_desc%domain_size = domain_desc%total_size
        field_desc%levels = domain_desc%levels
        field_desc%field = c_loc(field)
    end subroutine ghex_unstruct_field_desc_init

    subroutine ghex_unstruct_pattern_setup(pattern, domain_descs)
        type(ghex_unstruct_pattern) :: pattern
        type(ghex_unstruct_domain_desc), dimension(:), target :: domain_descs
        call ghex_unstruct_pattern_setup_impl(pattern, c_loc(domain_descs), size(domain_descs))
    end subroutine ghex_unstruct_pattern_setup

    subroutine ghex_unstruct_domain_desc_clear(domain_desc)
        type(ghex_unstruct_domain_desc) :: domain_desc

        domain_desc%id = -1
        domain_desc%vertices = c_null_ptr
        domain_desc%total_size = 0
        domain_desc%outer_indices = c_null_ptr
        domain_desc%outer_size = 0
        domain_desc%levels = 1
    end subroutine ghex_unstruct_domain_desc_clear

    subroutine ghex_unstruct_field_desc_clear(field_desc)
        type(ghex_unstruct_field_desc) :: field_desc

        field_desc%domain_id = -1
        field_desc%domain_size = 0
        field_desc%levels = 1
        field_desc%field = c_null_ptr
    end subroutine ghex_unstruct_field_desc_clear

END MODULE ghex_unstructured_mod
