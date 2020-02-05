PROGRAM test_halo_exchange
  use omp_lib
  use ghex_mod
  use ghex_structured_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size
  integer :: tmp
  integer :: halo(6)
  integer :: l_dim(3) = [32, 32, 32]
  type(ghex_domain_descriptor), target, dimension(:) :: domain_desc(1)
  type(ghex_domain_descriptor), pointer, dimension(:) :: pdomains
  integer :: local_extents(3)
  integer :: gfirst(3), glast(3)
  integer :: gdim(3) = [3, 3, 3]
  type(ghex_pattern) :: pattern
  real(8), dimension(:,:,:), pointer :: data
  type(ghex_field_descriptor) :: field_desc
  type(ghex_communication_object) :: co
  type(ghex_exchange_descriptor) :: ex_desc
  type(ghex_exchange_handle) :: hex

  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
  call mpi_comm_rank(mpi_comm_world, rank, mpi_err)
  call mpi_comm_size(mpi_comm_world, size, mpi_err)

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world)
  
  ! setup
  halo = [2,2,2,2,2,2]
  
  ! use 27 ranks to test all halo connectivities
  if (size /= 27) then
     print *, "Usage: this test must be executed with 27 mpi ranks"
     call exit(1)
  end if

  ! define the global index domain
  gfirst = [1, 1, 1]
  glast = gdim * l_dim

  ! define the local index domain
  domain_desc(1)%id = rank
  domain_desc(1)%device_id = DeviceCPU
  tmp = rank
  domain_desc(1)%first(3) = modulo(tmp, 3)
  tmp = tmp/3
  domain_desc(1)%first(1) = modulo(tmp, 3)
  tmp = tmp/3
  domain_desc(1)%first(2) = modulo(tmp, 3)
  domain_desc(1)%first = domain_desc(1)%first * l_dim + 1
  domain_desc(1)%last = domain_desc(1)%first + l_dim - 1
  domain_desc(1)%gfirst = gfirst
  domain_desc(1)%glast = glast
  
  ! allocate field data
  local_extents = [l_dim(3)+halo(5)+halo(6), l_dim(2)+halo(3)+halo(4), l_dim(1)+halo(1)+halo(2)]
  allocate(data(local_extents(3), local_extents(2), local_extents(1)))
  data(:,:,:) = rank

  ! initialize the field datastructure
  call ghex_field_init(field_desc, data, halo)

  ! add the field to the local domain
  call ghex_domain_add_field(domain_desc(1), field_desc)

  ! compute the halo information for all domains and fields
  ex_desc = ghex_exchange_new(domain_desc)

  ! create communication object
  co = ghex_struct_co_new()

  ! exchange halos
  ! hex = ghex_struct_exchange(co, fields)

  ! cleanup
  call ghex_exchange_delete(ex_desc)
  call ghex_struct_co_delete(co)
  call ghex_domain_delete(domain_desc(1))
  call ghex_finalize()
  call mpi_finalize(mpi_err)

END PROGRAM test_halo_exchange
