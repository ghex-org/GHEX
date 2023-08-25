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
  use iso_c_binding
  use ghex_mod
  use ghex_structured_mod

  implicit none

  include 'mpif.h'

  character(len=512) :: arg
  real    :: tic, toc
  integer :: mpi_err, mpi_threading, it
  integer :: nthreads = 1
  integer :: world_size, world_rank

  integer :: gfirst(3), glast(3)       ! global index space
  integer ::  first(3),  last(3)       ! local index space

  integer :: cart_dim(3) = [0, 0, 0]
  integer :: rank_coord(3)             ! local rank coordinates in a cartesian rank space
  integer :: halo(6)                   ! halo definition

  integer :: xsb, xeb, ysb, yeb, zsb, zeb
  integer :: xs , xe , ys , ye , zs , ze
  integer :: comm_cart

  ! parameters
  integer :: mb = 5                    ! halo width
  integer :: niters = 10
  integer :: ldim(3) = [64, 64, 64]    ! dimensions of the local domains
  integer :: nfields = 8
  logical :: periodic(3) = [.true.,.true.,.true.] ! for MPI_Cart_create
  integer, parameter :: nfields_max = 8

  ! data field pointers
  type hptr
     real(kind=4), dimension(:,:,:), pointer :: ptr
  end type hptr
  type(hptr) :: data_ptr(nfields_max)

  ! GHEX stuff
  type(ghex_struct_field)                :: field_desc   ! field descriptor

  ! single domain, multiple fields
  type(ghex_struct_domain)               :: domain_desc  ! domain descriptor
  type(ghex_struct_communication_object) :: co           ! communication object
  type(ghex_struct_exchange_descriptor)  :: ed           ! exchange descriptor
  type(ghex_struct_exchange_handle)      :: eh           ! exchange handle

  procedure(f_cart_rank_neighbor), pointer :: p_cart_nbor
  p_cart_nbor => cart_rank_neighbor

  ! init mpi
  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
  call mpi_comm_rank(mpi_comm_world, world_rank, mpi_err)
  call mpi_comm_size(mpi_comm_world, world_size, mpi_err)

  ! Cartesian communicator decomposition
  call mpi_dims_create(world_size, 3, cart_dim, mpi_err)

  if (world_rank == 0) then
    ! check if correct number of ranks
    if (product(cart_dim) /= world_size) then
      write (*,"(a, i4, a, i4, a)") "Number of ranks (", world_size, ") doesn't match the domain decomposition (", product(cart_dim), ")"
      call exit(1)
    end if
  end if

  ! cartesian communicator
  call mpi_cart_create(mpi_comm_world, 3, cart_dim, periodic, .true., comm_cart, mpi_err)

  ! local indices in the rank index space
  call mpi_comm_rank(comm_cart, world_rank, mpi_err)
  call mpi_cart_coords(comm_cart, world_rank, 3, rank_coord, mpi_err)

  ! halo information
  halo(:) = 0
  halo(1:2) = mb
  halo(3:4) = mb
  halo(5:6) = mb

  if (world_rank==0) then
    print *, "rank cartesian grid:  ", cart_dim
    print *, "per-rank domain size: ", ldim
    print *, "halos:                ", halo
  end if

  ! define the global index domain
  gfirst = [1, 1, 1]
  glast = cart_dim * ldim

  ! define the local domain
  first = (rank_coord) * ldim + 1
  last  = first + ldim - 1

  ! helper variables - local index ranges
  xs  = first(1)
  xe  = last(1)
  ys  = first(2)
  ye  = last(2)
  zs  = first(3)
  ze  = last(3)

  xsb = xs - halo(1)
  xeb = xe + halo(2)
  ysb = ys - halo(3)
  yeb = ye + halo(4)
  zsb = zs - halo(5)
  zeb = ze + halo(6)

  ! allocate and initialize data cubes
  it = 1
  do while (it <= nfields)
    allocate(data_ptr(it)%ptr(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
    data_ptr(it)%ptr(xs:xe, ys:ye, zs:ze) = world_rank+it
    it = it+1
  end do

  ! ------ GHEX
  call ghex_init(nthreads, comm_cart)

  ! create communication object
  call ghex_co_init(co)

  ! create domain description
  call ghex_domain_init(domain_desc, world_rank, first, last, gfirst, glast, p_cart_nbor)

  ! initialize the field datastructure
  it = 1
  do while (it <= nfields)
    call ghex_field_init(field_desc, data_ptr(it)%ptr, halo, periodic=periodic)
    call ghex_domain_add_field(domain_desc, field_desc)
    call ghex_free(field_desc)
    it = it+1
  end do

  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

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
  call test_exchange(data_ptr, rank_coord)

  ! cleanup
  call ghex_free(co)
  call ghex_free(domain_desc)
  call ghex_free(ed)
  call ghex_finalize()
  call mpi_finalize(mpi_err)
  call exit(0)

contains

  subroutine test_exchange(data_ptr, rank_coord)
    type(hptr), intent(in), dimension(:) :: data_ptr
    integer, intent(in) :: rank_coord(3)             ! local rank coordinates in a cartesian rank space
    integer :: nbor_coord(3), nbor_rank, ix, iy, iz, jx, jy, jz
    real(kind=4), dimension(:,:,:), pointer :: data
    integer :: isx(-1:1), iex(-1:1)
    integer :: isy(-1:1), iey(-1:1)
    integer :: isz(-1:1), iez(-1:1)
    integer :: i, j, k, did
    logical :: err

    err = .false.
    isx = (/xsb, xs, xe+1/);
    iex = (/xs-1, xe, xeb/);

    isy = (/ysb, ys, ye+1/);
    iey = (/ys-1, ye, yeb/);

    isz = (/zsb, zs, ze+1/);
    iez = (/zs-1, ze, zeb/);

    did = 1
    do while(did<=nfields)
      i = -1
      do while (i<=1)
        j = -1
        do while (j<=1)
          k = -1
          do while (k<=1)

            ! get nbor rank
            nbor_coord = rank_coord + (/i,j,k/);
            call mpi_cart_rank(comm_cart, nbor_coord, nbor_rank, mpi_err)

            ! check cube values
            if (.not.all(data_ptr(did)%ptr(isx(i):iex(i), isy(j):iey(j), isz(k):iez(k))==nbor_rank+did)) then
              print *, "wrong halo ", i, j, k
              err = .true.
            end if
            k = k+1
          end do
          j = j+1
        end do
        i = i+1
      end do
      did = did+1
    end do
    if (err) then
      call exit(1)
    end if
  end subroutine test_exchange

  subroutine cart_rank_neighbor (id, offset_x, offset_y, offset_z, nbid_out, nbrank_out) bind(c)
    use iso_c_binding
    integer(c_int), value, intent(in) :: id, offset_x, offset_y, offset_z
    integer(c_int), intent(out) :: nbid_out, nbrank_out
    integer :: coord(3)

    call mpi_cart_coords(comm_cart, id, 3, coord, mpi_err);
    coord(1) = coord(1) + offset_x;
    coord(2) = coord(2) + offset_y;
    coord(3) = coord(3) + offset_z;
    call mpi_cart_rank(comm_cart, coord, nbrank_out, mpi_err);
    nbid_out = nbrank_out;
  end subroutine cart_rank_neighbor

END PROGRAM test_halo_exchange
