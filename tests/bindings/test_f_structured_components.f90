PROGRAM test_halo_exchange
  use omp_lib
  use ghex_mod

  implicit none  

  include 'mpif.h'  

  character(len=512) :: arg
  real    :: tic, toc
  integer :: ierr, mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size, world_rank
  integer :: tmp, i, it
  integer :: gfirst(3), glast(3)       ! global index space
  integer :: first(3), last(3)
  integer :: gdim(3) = [1, 1, 1]       ! number of domains
  integer :: ldim(3) = [128, 128, 128] ! dimensions of the local domains
  integer :: rank_coord(3)             ! local rank coordinates in a cartesian rank space
  integer :: halo(6)                   ! halo definition
  integer :: mb = 5                    ! halo width
  integer :: niters = 100
  integer, parameter :: ncomponents = 8

  integer :: xsb, xeb, ysb, yeb, zsb, zeb
  integer :: xs , xe , ys , ye , zs , ze
  integer :: xr , xrb, yr , yrb, zr , zrb
  integer :: C_CART

  type hptr
     real(ghex_fp_kind), dimension(:,:,:,:), pointer :: ptr
  end type hptr
  
  ! exchange 1 data field with multiple components (vector)
  type(hptr) :: data_ptr

  ! GHEX stuff
  type(ghex_communicator)                :: comm         ! communicator
  type(ghex_struct_field)                :: field_desc

  ! single domain, multiple fields
  type(ghex_struct_domain)               :: domain_desc
  type(ghex_struct_communication_object) :: co
  type(ghex_struct_exchange_descriptor)  :: ed
  type(ghex_struct_exchange_handle)      :: eh

  if (command_argument_count() /= 6) then
     print *, "Usage: <benchmark> [grid size] [niters] [halo size] [rank dims :3] "
     call exit(1)
  end if

  ! domain grid dimensions
  call get_command_argument(1, arg)
  read(arg,*) ldim(1)
  ldim(2) = ldim(1)
  ldim(3) = ldim(1)

  ! number of iterations
  call get_command_argument(2, arg)
  read(arg,*) niters

  ! halo size
  call get_command_argument(3, arg)
  read(arg,*) mb

  ! rank grid dimensions
  call get_command_argument(4, arg)
  read(arg,*) gdim(1)
  call get_command_argument(5, arg)
  read(arg,*) gdim(2)
  call get_command_argument(6, arg)
  read(arg,*) gdim(3)

  ! init mpi
  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
  call mpi_comm_rank(mpi_comm_world, world_rank, mpi_err)
  call mpi_comm_size(mpi_comm_world, size, mpi_err)
  ! call mpi_dims_create(size, 3, gdim, mpi_err)
  ! call mpi_cart_create(mpi_comm_world, 3, gdim, [1, 1, 1], .true., C_CART,ierr)
  C_CART = mpi_comm_world

  call mpi_comm_rank(C_CART, rank, mpi_err)

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world)
  
  ! create ghex communicator
  comm = ghex_comm_new()

  ! halo information
  halo(:) = 0
  halo(1:2) = mb
  halo(3:4) = mb
  halo(5:6) = mb

  if (rank==0) then
     print *, "halos: ", halo
  end if
  
  ! use 27 ranks to test all halo connectivities
  if (size /= product(gdim)) then
    print *, "Usage: this test must be executed with ", product(gdim), " mpi ranks"
    call exit(1)
  end if

  ! define the global index domain
  gfirst = [1, 1, 1]
  glast = gdim * ldim

  ! local indices in the rank index space
  call rank2coord(rank, rank_coord)

  ! define the local domain
  first = (rank_coord-1) * ldim + 1
  last  = first + ldim - 1

  ! define local index ranges
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

  ! ! create communication object
  call ghex_co_init(co, comm)

  ! ---- field last ----
  ! allocate and initialize data cubes
  allocate(data_ptr%ptr(xsb:xeb, ysb:yeb, zsb:zeb, ncomponents), source=-1.0)
  data_ptr%ptr(xs:xe, ys:ye, zs:ze, :) = rank

  ! initialize the domain and field datastructures
  call ghex_domain_init(domain_desc, rank, first, last, gfirst, glast)
  call ghex_field_init(field_desc, data_ptr%ptr, halo, periodic=[1,1,1], layout=LayoutFieldLast)
  call ghex_domain_add_field(domain_desc, field_desc)
  call ghex_free(field_desc)

  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

  ! exchange halos
  eh = ghex_exchange(co, ed)
  call ghex_wait(eh)
  call cpu_time(tic)
  it = 0
  do while (it < niters)
    eh = ghex_exchange(co, ed)
    call ghex_wait(eh)
    it = it+1
  end do
  call cpu_time(toc)
  if (rank == 0) then 
     print *, rank, " exchange compact:      ", (toc-tic)
  end if
  call ghex_free(ed)
  call ghex_free(domain_desc) 
  deallocate(data_ptr%ptr)

  ! ! ---- field first ----
  ! ! allocate and initialize data cubes
  ! allocate(data_ptr%ptr(ncomponents, xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  ! data_ptr%ptr(:, xs:xe, ys:ye, zs:ze) = rank

  ! ! initialize the domain and field datastructures
  ! call ghex_domain_init(domain_desc, rank, first, last, gfirst, glast)
  ! call ghex_field_init(field_desc, data_ptr%ptr, halo, periodic=[1,1,1], layout=LayoutFieldFirst)
  ! call ghex_domain_add_field(domain_desc, field_desc)
  ! call ghex_free(field_desc)

  ! ! compute the halo information for all domains and fields
  ! ed = ghex_exchange_desc_new(domain_desc)

  ! ! exchange halos
  ! eh = ghex_exchange(co, ed)
  ! call ghex_wait(eh)
  ! call cpu_time(tic)
  ! it = 0
  ! do while (it < niters)
  !   eh = ghex_exchange(co, ed)
  !   call ghex_wait(eh)
  !   it = it+1
  ! end do
  ! call cpu_time(toc)
  ! if (rank == 0) then 
  !    print *, rank, " exchange compact:      ", (toc-tic)
  ! end if
  ! call ghex_free(ed)
  ! call ghex_free(domain_desc)
  ! deallocate(data_ptr%ptr)

  ! cleanup 
  call ghex_free(co)  
  call ghex_finalize()
  call mpi_finalize(mpi_err)

contains
  
  ! -------------------------------------------------------------
  ! cartesian coordinates computations
  ! -------------------------------------------------------------
  subroutine rank2coord(rank, coord)
    integer :: rank, tmp, ierr
    integer :: coord(3)

    if (C_CART == mpi_comm_world) then
       tmp = rank;        coord(1) = modulo(tmp, gdim(1))
       tmp = tmp/gdim(1); coord(2) = modulo(tmp, gdim(2))
       tmp = tmp/gdim(2); coord(3) = tmp
    else
       call mpi_cart_coords(C_CART, rank, 3, coord, ierr)
    end if
    coord = coord + 1
  end subroutine rank2coord

END PROGRAM
