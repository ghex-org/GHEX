!
! ghex-org
!
! Copyright (c) 2014-2023, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause
!
PROGRAM test_halo_exchange
    use iso_c_binding ! TO DO: should not be needed here
    use ghex_mod ! ghex_init() etc.
    use ghex_unstructured_mod ! ghex unstruct bindings

    implicit none

    include 'mpif.h'

    ! utils
    integer :: mpi_err, world_rank, world_size, it
    real :: tic, toc
    logical :: passed = .true.

    ! parameters
    integer, parameter :: niters = 1
    integer, parameter :: ndomains = 1
    integer, parameter :: nfields = 4

    ! application domain
    ! TO DO: the test application is currently implemented assuming 1 domain per rank,
    ! despite GHEX built-in support for multiple domains per rank.
    ! For tests with multiple domains a minor refactoring is needed.
    integer, dimension(:), pointer :: vertices
    integer, dimension(:), pointer :: outer_indices

    ! application field pointers
    type hptr
        real(ghex_fp_kind), dimension(:,:), pointer :: ptr
    end type hptr
    type(hptr) :: field_ptrs(nfields)

    ! GHEX types
    type(ghex_unstruct_domain_desc), dimension(:), pointer :: domain_descs ! domain descriptors
    type(ghex_unstruct_pattern) :: pattern ! pattern
    type(ghex_unstruct_field_desc) :: field_descs(nfields) ! field descriptors
    type(ghex_unstruct_communication_object) :: co ! communication object
    type(ghex_unstruct_exchange_args) :: args ! exchange arguments
    type(ghex_unstruct_exchange_handle) :: h ! exchange handle

    ! init mpi
    call mpi_init(mpi_err) ! TO DO: mpi_init_thread
    call mpi_comm_rank(mpi_comm_world, world_rank, mpi_err)
    call mpi_comm_size(mpi_comm_world, world_size, mpi_err)

    ! init GHEX
    call ghex_init(1, mpi_comm_world) ! TO DO: multithread

    ! application domain decomposition
    call domain_decompose(world_rank, vertices, outer_indices)

    ! init GHEX domain descriptors
    allocate(domain_descs(ndomains))
    do it = 1, ndomains
        call domain_desc_init(domain_descs(it), world_rank, vertices, outer_indices)
    end do

    ! setup GHEX pattern
    call ghex_unstruct_pattern_setup(pattern, domain_descs)

    ! application data
    ! TO DO: all fields are defined on the first (and only) domain
    do it = 1, nfields
        call field_init(field_ptrs(it), it, domain_descs(1), vertices)
    end do

    ! init GHEX field descriptors
    ! TO DO: all fields are defined on the first (and only) domain
    do it = 1, nfields
        call ghex_unstruct_field_desc_init(field_descs(it), domain_descs(1), field_ptrs(it)%ptr)
    end do

    ! init GHEX communication object
    call ghex_unstruct_communication_object_init(co)

    ! init GHEX exchange args
    call ghex_unstruct_exchange_args_init(args)

    ! add pattern-field matches to GHEX exchange args
    do it = 1, nfields
        call ghex_unstruct_exchange_args_add(args, pattern, field_descs(it))
    end do

    ! exchange loop
    call cpu_time(tic)
    do it = 1, niters
        h = ghex_unstruct_exchange(co, args)
        call ghex_unstruct_exchange_handle_wait(h)
        call ghex_free(h)
    end do
#ifdef GHEX_ENABLE_BARRIER
    call ghex_barrier(GhexBarrierGlobal)
#endif
    call cpu_time(toc)

    ! validate results
    ! TO DO: all fields are defined on the first (and only) domain
    do it = 1, nfields
        passed = passed .and. field_check(field_ptrs(it), it, domain_descs(1), vertices)
    end do

    ! print results
    if (passed) then
        print *, "rank ", world_rank, ": PASSED"
        if (world_rank == 0) then
            print *, "exchange time for rank 0: ", (toc-tic)
        end if
    else
        print *, "rank ", world_rank, ": FAILED"
    end if

    ! cleanup GHEX
    call ghex_free(args)
    call ghex_free(co)
    call ghex_free(pattern)
    do it = 1, nfields
        call ghex_clear(field_descs(it))
    end do
    do it = 1, ndomains
        call ghex_clear(domain_descs(it))
    end do
    deallocate(domain_descs)

    ! cleanup application
    do it = 1, nfields
        deallocate(field_ptrs(it)%ptr)
    end do
    deallocate(vertices)

    ! finalize GHEX
    call ghex_finalize()

    ! finalize mpi
    call mpi_finalize(mpi_err)

    if (passed) then
        call exit(0)
    else
        call exit(1)
    end if

contains

    subroutine domain_decompose(id, vertices, outer_indices)
        integer :: id
        integer, dimension(:), pointer :: vertices ! TO DO: target?
        integer, dimension(:), pointer :: outer_indices ! TO DO: target?

        select case(id)
        case(0)
            allocate(vertices(9))
            vertices = (/0, 13, 5, 2, 1, 3, 7, 11, 20/)
            allocate(outer_indices(5))
            outer_indices = (/4, 5, 6, 7, 8/)
        case(1)
            allocate(vertices(11))
            vertices = (/1, 19, 20, 4, 7, 15, 8, 0, 9, 13, 16/)
            allocate(outer_indices(4))
            outer_indices = (/7, 8, 9, 10/)
        case(2)
            allocate(vertices(6))
            vertices = (/3, 16, 18, 1, 5, 6/)
            allocate(outer_indices(3))
            outer_indices = (/3, 4, 5/)
        case(3)
            allocate(vertices(9))
            vertices = (/17, 6, 11, 10, 12, 9, 0, 3, 4/)
            allocate(outer_indices(3))
            outer_indices = (/6, 7, 8/)
        endselect
    end subroutine domain_decompose

    subroutine domain_desc_init(domain_desc, id, vertices, outer_indices)
        type(ghex_unstruct_domain_desc) :: domain_desc
        integer :: id
        integer, dimension(:), target :: vertices ! TO DO: pointer?
        integer, dimension(:), target :: outer_indices 

        select case(id)
        case(0)
            call ghex_unstruct_domain_desc_init(domain_desc, 0, vertices, 9, outer_indices, 5, 10) ! 10 vertical layers
        case(1)
            call ghex_unstruct_domain_desc_init(domain_desc, 1, vertices, 11, outer_indices, 4, 10) ! 10 vertical layers
        case(2)
            call ghex_unstruct_domain_desc_init(domain_desc, 2, vertices, 6, outer_indices, 3, 10) ! 10 vertical layers
        case(3)
            call ghex_unstruct_domain_desc_init(domain_desc, 3, vertices, 9, outer_indices, 3, 10) ! 10 vertical layers
        endselect
    end subroutine domain_desc_init

    subroutine field_init(field_ptr, field_id, domain_desc, vertices)
        type(hptr) :: field_ptr
        integer :: field_id
        type(ghex_unstruct_domain_desc) :: domain_desc
        integer, dimension(:), target :: vertices ! TO DO: pointer?
        integer :: i, j
        real(ghex_fp_kind) :: part_1, part_2

        allocate(field_ptr%ptr(domain_desc%levels, domain_desc%total_size), source=-1.)
        ! part_1 = domain_id * 10000 + field_id * 1000
        part_1 = (field_id - 1) * 1000 ! TO DO: domain_id temporarily removed for simplicity
        do j = 1, domain_desc%total_size - domain_desc%outer_size
            part_2 = part_1 + vertices(j) * 10
            do i = 1, domain_desc%levels
                field_ptr%ptr(i, j) = part_2 + i - 1
            end do
        end do
    end subroutine field_init

    logical function field_check(field_ptr, field_id, domain_desc, vertices)
        type(hptr) :: field_ptr
        integer :: field_id
        type(ghex_unstruct_domain_desc) :: domain_desc
        integer, dimension(:), target :: vertices ! TO DO: pointer?
        integer :: i, j
        real(ghex_fp_kind) :: part_1, part_2
        logical :: passed = .true.

        part_1 = (field_id - 1) * 1000 ! TO DO: domain_id temporarily removed for simplicity
        do j = domain_desc%total_size - domain_desc%outer_size + 1, domain_desc%total_size
            part_2 = part_1 + vertices(j) * 10
            do i = 1, domain_desc%levels
                passed = passed .and. ((field_ptr%ptr(i, j)) == (part_2 + i - 1))
            end do
        end do

        field_check = passed
    end function field_check

END PROGRAM test_halo_exchange
