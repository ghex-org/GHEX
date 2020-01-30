PROGRAM test_context
  use omp_lib
  use ghex_context_mod
  use ghex_comm_mod
  use ghex_structured_mod

  implicit none  

  include 'mpif.h'  

  type(ghex_context) :: context
  type(ghex_communicator) :: comm
  integer :: mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size
  integer :: tmp
  integer :: halo(6)
  integer :: l_dim(3) = [32, 32, 32]
  type(ghex_domain_descriptor), target, dimension(:) :: domain_desc(1)
  type(ghex_domain_descriptor), pointer, dimension(:) :: pdomains
  integer :: l_ext_buffer(3)
  integer :: g_first(3), g_last(3)
  integer :: g_dim(3) = [3, 3, 3]
  integer :: periodic(3) = [1, 1, 1]
  real(8), dimension(:,:,:), allocatable :: data
  type(ghex_pattern) :: pattern

  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)

  ! create a context object
  context = context_new(nthreads, mpi_comm_world)
  rank = context_rank(context)
  size = context_size(context)
  
  ! setup
  halo = [2,2,2,2,2,2]
  
  ! use 27 ranks to test all halo connectivities
  if (size /= 27) then
     print *, "Usage: this test must be executed with 27 mpi ranks"
     call exit(1)
  end if

  ! global index domain
  g_first = [1, 1, 1]
  g_last = g_dim * l_dim

  ! local index domain
  domain_desc(1)%id = rank  
  tmp = rank
  domain_desc(1)%first(3) = modulo(tmp, 3)
  tmp = tmp/3
  domain_desc(1)%first(1) = modulo(tmp, 3)
  tmp = tmp/3
  domain_desc(1)%first(2) = modulo(tmp, 3)
  domain_desc(1)%first = domain_desc(1)%first * l_dim + 1
  domain_desc(1)%last = domain_desc(1)%first + l_dim - 1

  print *, rank, domain_desc(1)%first, domain_desc(1)%last

  ! allocate data
  l_ext_buffer = [l_dim(2)+halo(3)+halo(4), l_dim(1)+halo(1)+halo(2), l_dim(3)+halo(5)+halo(6)]
  allocate(data(l_ext_buffer(1), l_ext_buffer(2), l_ext_buffer(3)))

  ! make pattern
  pdomains => domain_desc
  pattern = ghex_make_pattern(context, halo, pdomains, periodic, g_first, g_last)

  ! delete the ghex context
  call context_delete(context)  
  call mpi_finalize(mpi_err)

END PROGRAM test_context
