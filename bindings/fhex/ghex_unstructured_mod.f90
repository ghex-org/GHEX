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

    ! field descriptor
    type, bind(c) :: ghex_unstruct_field_desc
        type(c_ptr) :: domain = c_null_ptr
        type(c_ptr) :: field = c_null_ptr ! OK
        integer(c_int) :: field_size = 0 ! From HERE (included)
        integer(c_int) :: device_id = GhexDeviceUnknown
    end type ghex_unstruct_field

    ! To HERE

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

END MODULE ghex_unstructured_mod
