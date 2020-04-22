PROGRAM test_f_cubed_sphere
  use ghex_mod

  implicit none

  include 'mpif.h'

  real    :: tic, toc
  integer :: ierr, mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size, world_rank
  integer :: tmp, i, j
  integer :: ntiles = 6
  integer :: tile_dims(2) = [2, 2]         ! each tile is split into tile_dims ranks, in X and Y dimensions
  integer :: tile = -1, tile_coord(2)      ! which tile do we belong to, and what are our tile coordinates
  integer :: cube(2) = [10, 6]             ! dimensions of the tile domains, (nx*nx*ndepth)
  integer :: blkx, blky
  integer :: halo(4), mb                   ! halo definition
  integer :: first(2), last(2)

  ! exchange scalar and vector fields
  real(ghex_fp_kind), dimension(:,:,:), pointer :: data_scalar
  real(ghex_fp_kind), dimension(:,:,:,:), pointer :: data_vector
  integer :: n_components = 3

  ! GHEX stuff
  type(ghex_cubed_sphere_domain)               :: domain_desc
  type(ghex_cubed_sphere_field)                :: field_desc
  type(ghex_cubed_sphere_communication_object) :: co
  type(ghex_cubed_sphere_exchange_descriptor)  :: ed
  type(ghex_cubed_sphere_exchange_handle)      :: eh

  ! init mpi
  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
  call mpi_comm_rank(mpi_comm_world, world_rank, mpi_err)
  call mpi_comm_size(mpi_comm_world, size, mpi_err)
  call mpi_comm_rank(mpi_comm_world, rank, mpi_err)

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world)

  ! halo width
  halo(:) = mb

  ! check if we have the right number of ranks
  if (size /= product(tile_dims)*ntiles) then
    print *, "Usage: this test must be executed with ", product(tile_dims)*ntiles, " mpi ranks"
    call exit(1)
  end if

  ! rank to tile
  tile = modulo(rank, ntiles)
  tmp = rank/ntiles;      tile_coord(2) = modulo(tmp, tile_dims(2))
  tmp = tmp/tile_dims(2); tile_coord(1) = tmp
  tile_coord = tile_coord + 1
  blkx = cube(1)/tile_dims(1)
  blky = cube(1)/tile_dims(2)
  if (blkx*tile_dims(1) /= cube(1) .or. blkx*tile_dims(1) /= cube(1)) then
    print *, "The per-tile grid dimensions are not divisible by the rank dimensions."
    call exit(1)
  end if

  ! define the local domain
  first = [(tile_coord(1)-1)*blkx+1, (tile_coord(2)-1)*blky+1]
  last  = first + [blkx, blky] - 1
  print *, "rank ", rank, " tile ", tile, " first ", first, " last ", last
   
  ! scalar field exchange
  call ghex_cubed_sphere_domain_init(domain_desc, tile, cube, first, last)  
  allocate(data_scalar(blkx+sum(halo(1:2)), blky+sum(halo(3:4)), cube(2)), source=-1.0)
  data_scalar(:, :, :) = rank
  call ghex_field_init(field_desc, data_scalar, halo)
  call ghex_domain_add_field(domain_desc, field_desc)
  call ghex_free(field_desc)

  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

  ! create communication object
  call ghex_co_init(co)

  ! exchange halos
  eh = ghex_exchange(co, ed)
  call ghex_wait(eh)

  ! cleanups
  call ghex_free(domain_desc)
  call ghex_free(ed)
  
  ! vector field exchange
  call ghex_cubed_sphere_domain_init(domain_desc, tile, cube, first, last)  
  allocate(data_vector(blkx+sum(halo(1:2)), blky+sum(halo(3:4)), cube(2), n_components), source=-1.0)
  data_vector(:, :, :, 1) = 10*rank
  data_vector(:, :, :, 2) = 100*rank
  data_vector(:, :, :, 3) = 1000*rank
  call ghex_field_init(field_desc, data_vector, halo, n_components=n_components, is_vector=.true.)
  call ghex_domain_add_field(domain_desc, field_desc)
  call ghex_free(field_desc)

  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

  ! exchange halos
  eh = ghex_exchange(co, ed)
  call ghex_wait(eh) 

  ! cleanups
  call ghex_free(domain_desc)
  call ghex_free(ed)
  call ghex_finalize()
  call mpi_finalize(mpi_err)

END PROGRAM test_f_cubed_sphere
