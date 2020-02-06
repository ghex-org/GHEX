PROGRAM test_halo_exchange
  use omp_lib
  use ghex_mod
  use ghex_structured_mod
  use ghex_exchange_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size
  integer :: tmp
  integer :: gfirst(3), glast(3)  ! global index space
  integer :: gdim(3) = [3, 3, 3]  ! number of domains
  integer :: ldim(3) = [1, 1, 1]  ! dimensions of the local domains
  integer :: local_extents(3)     ! index space of the local domains, in global coordinates
  integer :: halo(6)              ! halo definition
  real(8), dimension(:,:,:), pointer :: data

  type(ghex_domain_descriptor), target, dimension(:) :: domain_desc(1)
  type(ghex_field_descriptor)     :: field_desc
  type(ghex_communication_object) :: co
  type(ghex_exchange_descriptor)  :: ex_desc
  type(ghex_exchange_handle)      :: ex_handle

  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
  call mpi_comm_rank(mpi_comm_world, rank, mpi_err)
  call mpi_comm_size(mpi_comm_world, size, mpi_err)

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world)
  
  ! setup
  halo = [1,1,1,1,1,1]
  
  ! use 27 ranks to test all halo connectivities
  if (size /= 27) then
     print *, "Usage: this test must be executed with 27 mpi ranks"
     call exit(1)
  end if

  ! define the global index domain
  gfirst = [1, 1, 1]
  glast = gdim * ldim

  ! define the local index domain
  domain_desc(1)%id = rank
  domain_desc(1)%device_id = DeviceCPU
  tmp = rank;  domain_desc(1)%first(1) = modulo(tmp, 3)
  tmp = tmp/3; domain_desc(1)%first(2) = modulo(tmp, 3)
  tmp = tmp/3; domain_desc(1)%first(3) = modulo(tmp, 3)
  domain_desc(1)%first = domain_desc(1)%first * ldim + 1
  domain_desc(1)%last  = domain_desc(1)%first + ldim - 1
  domain_desc(1)%gfirst = gfirst
  domain_desc(1)%glast  = glast

  ! print *, rank, domain_desc(1)%first, domain_desc(1)%last
  ! call exit
  
  ! allocate field data
  local_extents = [ldim(1)+halo(1)+halo(2), ldim(2)+halo(3)+halo(4), ldim(3)+halo(5)+halo(6)]
  allocate(data(local_extents(1), local_extents(2), local_extents(3)))
  data(:,:,:) = rank

  ! initialize the field datastructure
  call ghex_field_init(field_desc, data, halo, periodic=[0,0,0])

  ! add the field to the local domain
  call ghex_domain_add_field(domain_desc(1), field_desc)

  ! compute the halo information for all domains and fields
  ex_desc = ghex_exchange_desc_new(domain_desc)

  ! create communication object
  co = ghex_struct_co_new()

  ! exchange halos
  ex_handle = ghex_exchange(co, ex_desc)
  call ghex_wait(ex_handle)
  call ghex_delete(ex_handle)

  ! cleanup
  call ghex_delete(ex_desc)
  call ghex_delete(co)
  call ghex_delete(domain_desc(1))
  call ghex_finalize()
  call mpi_finalize(mpi_err)

END PROGRAM test_halo_exchange
