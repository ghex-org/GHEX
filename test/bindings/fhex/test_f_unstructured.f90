PROGRAM test_halo_exchange

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
    use ghex_mod
    use ghex_unstructured_mod

    implicit none

    include 'mpif.h'

    ! TO DO: mpi variables
    ! TO DO: nthreads
    ! TO DO: nfields

    ! data field pointers
    type hptr
        real(kind=4), dimension(:,:), pointer :: ptr
    end type hptr
    type(hptr) :: data_ptr(4) ! TO DO data_ptr(nfields_max)

    ! GHEX stuff
    type(ghex_struct_domain)               :: domain_desc  ! domain descriptor
    type(ghex_struct_field)                :: field_desc   ! field descriptor
    type(ghex_struct_communication_object) :: co           ! communication object
    type(ghex_struct_exchange_descriptor)  :: ed           ! exchange descriptor
    type(ghex_struct_exchange_handle)      :: eh           ! exchange handle

    ! init mpi
    call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
    call mpi_comm_rank(mpi_comm_world, world_rank, mpi_err)
    call mpi_comm_size(mpi_comm_world, world_size, mpi_err)

    ! allocate and initialize data
    it = 1
    do while (it <= nfields)
    ! allocate(data_ptr(it)%ptr(...), source=-1.0)
    ! data_ptr(it)%ptr(...) = ...
    it = it+1
    end do

    ! ------ GHEX
    call ghex_init(nthreads, mpi_comm_world)

    ! create domain description
    call ghex_domain_init(domain_desc, ...)

    ! initialize the field datastructure
    it = 1
    do while (it <= nfields)
    call ghex_field_init(field_desc, ...)
    call ghex_exchange_desc_add_field(domain_desc, field_desc)
    call ghex_free(field_desc)
    it = it+1
    end do

    ! compute the halo information for all domains and fields
    ed = ghex_exchange_desc_new(domain_desc, field_desc)

    ! create communication object
    call ghex_co_init(co)

    ! exchange loop
    it = 0
    call cpu_time(tic)
    do while (it < niters)
    eh = ghex_exchange(co, ed)
    call ghex_wait(eh)
    call ghex_free(eh)
    it = it+1
    end do
#ifdef GHEX_ENABLE_BARRIER
    call ghex_barrier(GhexBarrierGlobal)
#endif
    call cpu_time(toc)
    if (world_rank == 0) then
        print *, "exchange time:      ", (toc-tic)
    end if

    ! validate the results
    ! call test_exchange(data_ptr, rank_coord)

    ! cleanup
    call ghex_free(co)
    call ghex_free(domain_desc)
    call ghex_free(ed)
    call ghex_finalize()
    call mpi_finalize(mpi_err)
    call exit(0)

END PROGRAM test_halo_exchange
