PROGRAM test_halo_exchange
  use omp_lib
  use ghex_mod

  implicit none  

  include 'mpif.h'  

  real    :: tic, toc
  integer :: ierr, mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size, world_rank
  integer :: tmp, i, it
  integer :: gfirst(3), glast(3)       ! global index space
  integer :: first(3), last(3)
  integer :: gdim(3) = [2, 4, 2]       ! number of domains
  integer :: ldim(3) = [64, 64, 64]    ! dimensions of the local domains
  integer :: rank_coord(3)             ! local rank coordinates in a cartesian rank space
  integer :: halo(6)                   ! halo definition
  integer :: niters = 100

  integer :: xsb, xeb, ysb, yeb, zsb, zeb
  integer :: xs , xe , ys , ye , zs , ze
  integer :: xr , xrb, yr , yrb, zr , zrb, mb
  integer :: C_CART

  type hptr
     real(ghex_fp_kind), dimension(:,:,:,:), pointer :: ptr
  end type hptr
  
  ! exchange 1 data field with multiple components (vector)
  type(hptr) :: data_ptr_fl ! field last
  type(hptr) :: data_ptr_ff ! field first

  ! GHEX stuff
  type(ghex_struct_field)                :: field_desc

  ! single domain, multiple fields
  type(ghex_struct_domain)               :: domain_desc
  type(ghex_struct_communication_object) :: co
  type(ghex_struct_exchange_descriptor)  :: ed
  type(ghex_struct_exchange_handle)      :: eh

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

  ! halo width
  mb = 1
  
  ! halo information
  halo(:) = 0
  if (gdim(1) >1) then
    halo(1:2) = mb
  end if
  if (gdim(2) >1) then
    halo(3:4) = mb
  end if
  if (gdim(3) >1) then
    halo(5:6) = mb
  end if

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

  ! allocate and initialize data cubes
  allocate(data_ptr_fl%ptr(xsb:xeb, ysb:yeb, zsb:zeb, 3), source=-1.0)
  data_ptr_fl%ptr(xs:xe, ys:ye, zs:ze, :) = rank
  allocate(data_ptr_ff%ptr(3, xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  data_ptr_ff%ptr(:, xs:xe, ys:ye, zs:ze) = rank

  ! ---- field last ----
  ! initialize the domain and field datastructures
  call ghex_domain_init(domain_desc, rank, first, last, gfirst, glast)
  call ghex_field_init(field_desc, data_ptr_fl%ptr, halo, periodic=[1,1,1], layout=LayoutFieldLast)
  call ghex_domain_add_field(domain_desc, field_desc)
  call ghex_free(field_desc)

  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

  ! create communication object
  call ghex_co_init(co)

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

  ! ---- field first ----
  ! initialize the domain and field datastructures
  call ghex_domain_init(domain_desc, rank, first, last, gfirst, glast)
  call ghex_field_init(field_desc, data_ptr_ff%ptr, halo, periodic=[1,1,1], layout=LayoutFieldFirst)
  call ghex_domain_add_field(domain_desc, field_desc)
  call ghex_free(field_desc)

  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

  ! create communication object
  call ghex_co_init(co)

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

  call mpi_barrier(mpi_comm_world, mpi_err)

  ! cleanup 
  call ghex_free(co)
  deallocate(data_ptr_fl%ptr)
  deallocate(data_ptr_ff%ptr)
  
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
       tmp = rank;        coord(3) = modulo(tmp, gdim(3))
       tmp = tmp/gdim(3); coord(2) = modulo(tmp, gdim(2))
       tmp = tmp/gdim(2); coord(1) = tmp
    else
       call mpi_cart_coords(C_CART, rank, 3, coord, ierr)
    end if
    coord = coord + 1
  end subroutine rank2coord

END PROGRAM
