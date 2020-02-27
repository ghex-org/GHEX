PROGRAM test_halo_exchange
  use omp_lib
  use ghex_mod
  use ghex_structured_mod
  use ghex_exchange_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size
  integer :: tmp, i
  integer :: gfirst(3), glast(3)  ! global index space
  integer :: gdim(3) = [1, 2, 2]  ! number of domains
  integer :: ldim(3) = [128, 64, 64]  ! dimensions of the local domains
  integer :: local_extents(3)     ! index space of the local domains, in global coordinates
  integer :: halo(6)              ! halo definition
  integer :: xsb, xeb, ysb, yeb, zsb, zeb
  integer :: xs , xe , ys , ye , zs , ze
  real(ghex_fp_kind), dimension(:,:,:), pointer :: data1, data2, data3, data4
  real(ghex_fp_kind), dimension(:,:,:), pointer :: data5, data6, data7, data8
  real :: tic, toc

  type(ghex_domain_descriptor), target, dimension(:) :: domain_desc(1), d1(1), d2(1), d3(1), d4(1), d5(1), d6(1), d7(1), d8(1)
  
  type(ghex_field_descriptor)     :: field_desc
  type(ghex_communication_object) :: co
  type(ghex_exchange_descriptor)  :: ed, ed1, ed2, ed3, ed4, ed5, ed6, ed7, ed8
  type(ghex_exchange_handle)      :: ex_handle

  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
  call mpi_comm_rank(mpi_comm_world, rank, mpi_err)
  call mpi_comm_size(mpi_comm_world, size, mpi_err)

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world)
  
  ! setup
  halo = [0,0,5,5,5,5]
  
  ! use 27 ranks to test all halo connectivities
  if (size /= product(gdim)) then
     print *, "Usage: this test must be executed with ", product(gdim), " mpi ranks"
     call exit(1)
  end if

  ! define the global index domain
  gfirst = [1, 1, 1]
  glast = gdim * ldim

  ! define the local index domain
  domain_desc(1)%id = rank
  domain_desc(1)%device_id = DeviceCPU
  tmp = rank;  domain_desc(1)%first(1) = modulo(tmp, gdim(1))
  tmp = tmp/gdim(1); domain_desc(1)%first(2) = modulo(tmp, gdim(2))
  tmp = tmp/gdim(2); domain_desc(1)%first(3) = modulo(tmp, gdim(3))
  domain_desc(1)%first = domain_desc(1)%first * ldim + 1
  domain_desc(1)%last  = domain_desc(1)%first + ldim - 1
  domain_desc(1)%gfirst = gfirst
  domain_desc(1)%glast  = glast

  call init_domain(d1, domain_desc)
  call init_domain(d2, domain_desc)
  call init_domain(d3, domain_desc)
  call init_domain(d4, domain_desc)
  call init_domain(d5, domain_desc)
  call init_domain(d6, domain_desc)
  call init_domain(d7, domain_desc)
  call init_domain(d8, domain_desc)
  
  ! define local index ranges
  xsb = domain_desc(1)%first(1) - halo(1)
  xeb = domain_desc(1)%last(1) + halo(2)
  ysb = domain_desc(1)%first(2) - halo(3)
  yeb = domain_desc(1)%last(2) + halo(4)
  zsb = domain_desc(1)%first(3) - halo(5)
  zeb = domain_desc(1)%last(3) + halo(6)

  xs  = domain_desc(1)%first(1)
  xe  = domain_desc(1)%last(1) 
  ys  = domain_desc(1)%first(2)
  ye  = domain_desc(1)%last(2) 
  zs  = domain_desc(1)%first(3)
  ze  = domain_desc(1)%last(3)
  
  ! allocate and initialize field data
  allocate(data1(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data2(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data3(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data4(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data5(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data6(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data7(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data8(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  
  data1(xs:xe, ys:ye, zs:ze) = rank

  ! initialize the field datastructure - COMPACT
  call ghex_field_init(field_desc, data1, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data2, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data3, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data4, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data5, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data6, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data7, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data8, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

  ! initialize the field datastructure - SEQUENCE  
  call ghex_field_init(field_desc, data1, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d1(1), field_desc)
  ed1 = ghex_exchange_desc_new(d1)

  call ghex_field_init(field_desc, data2, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d2(1), field_desc)
  ed2 = ghex_exchange_desc_new(d2)

  call ghex_field_init(field_desc, data3, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d3(1), field_desc)
  ed3 = ghex_exchange_desc_new(d3)

  call ghex_field_init(field_desc, data4, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d4(1), field_desc)
  ed4 = ghex_exchange_desc_new(d4)

  call ghex_field_init(field_desc, data5, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d5(1), field_desc)
  ed5 = ghex_exchange_desc_new(d5)

  call ghex_field_init(field_desc, data6, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d6(1), field_desc)
  ed6 = ghex_exchange_desc_new(d6)

  call ghex_field_init(field_desc, data7, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d7(1), field_desc)
  ed7 = ghex_exchange_desc_new(d7)

  call ghex_field_init(field_desc, data8, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d8(1), field_desc)
  ed8 = ghex_exchange_desc_new(d8)

  ! create communication object
  co = ghex_struct_co_new()

  ! exchange halos - COMPACT
  call cpu_time(tic)
  i = 0
  do while (i < 1000)
    ex_handle = ghex_exchange(co, ed)
    call ghex_wait(ex_handle)
    call ghex_delete(ex_handle)
    i = i+1
  end do
  call cpu_time(toc)
  print *, rank, " exchange compact:      ", (toc-tic)

  ! exchange halos - SEQUENCE
  call cpu_time(tic)
  i = 0
  do while (i < 1000)
    ex_handle = ghex_exchange(co, ed1); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed2); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed3); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed4); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed5); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed6); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed7); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed8); call ghex_wait(ex_handle)
    i = i+1
  end do
  call cpu_time(toc)
  print *, rank, " exchange compact:      ", (toc-tic)

  ! cleanup
  call ghex_delete(ed)
  call ghex_delete(co)
  call ghex_delete(domain_desc(1))
  call ghex_finalize()
  call mpi_finalize(mpi_err)

contains
  
  subroutine init_domain(dst, src)
    type(ghex_domain_descriptor), intent(inout) :: dst(1)
    type(ghex_domain_descriptor), intent(in) :: src(1)
    dst(1)%id = rank
    dst(1)%device_id = DeviceCPU
    dst(1)%first(:)  = src(1)%first(:)
    dst(1)%last(:)   = src(1)%last(:)
    dst(1)%gfirst(:) = src(1)%gfirst(:)
    dst(1)%glast(:)  = src(1)%glast(:)
  end subroutine init_domain
  
END PROGRAM test_halo_exchange
