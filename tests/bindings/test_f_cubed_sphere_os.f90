PROGRAM test_f_cubed_sphere
  use ghex_mod

  implicit none

  include 'mpif.h'

  real    :: tic, toc
  integer :: ierr, mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size, world_rank
  integer :: tmp, i, j, did
  integer :: ntiles = 6
  integer :: tile_dims(2) = [2, 2]         ! each tile is split into tile_dims ranks, in X and Y dimensions
  integer :: tile = -1, tile_coord(2)      ! which tile do we belong to, and what are our tile coordinates
  integer :: cube(2) = [10, 6]             ! dimensions of the tile domains, (nx*nx*ndepth)
  integer :: blkx, blky
  integer :: halo(4), mb                   ! halo definition
  integer :: niters = 100

  type hptr
     real(ghex_fp_kind), dimension(:,:,:), pointer :: ptr
  end type hptr
  
  ! exchange 8 data cubes
  real(ghex_fp_kind), dimension(:,:,:), pointer :: data_scalar
  type(hptr), dimension(:), pointer :: data_ptr
  real(ghex_fp_kind), dimension(:,:,:,:), pointer :: data_vector
  integer :: n_components = 3

  ! GHEX stuff
  type(ghex_cubed_sphere_domain), dimension(:), pointer :: domain_desc
  type(ghex_cubed_sphere_field)     :: field_desc
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
  mb = 1

  ! halo information
  halo(:) = mb

  ! check if we have the right number of ranks
  if (size /= ntiles) then
    print *, "Usage: this test must be executed with ", ntiles, " mpi ranks"
    call exit(1)
  end if

  ! allocate domains
  allocate(domain_desc(1:product(tile_dims)))
  
  ! store field data in a pointer array
  allocate(data_ptr(1:product(tile_dims)))
  
  ! rank to tile
  tile = rank
  did = 1
  i = 1
  do while (i <= tile_dims(1))
    j = 1
    do while (j <= tile_dims(2))
      tile_coord(1) = i
      tile_coord(2) = j

      blkx = cube(1)/tile_dims(1)
      blky = cube(1)/tile_dims(2)
      if (blkx*tile_dims(1) /= cube(1) .or. blkx*tile_dims(1) /= cube(1)) then
        print *, "The per-tile grid dimensions are not divisible by the rank dimensions."
        call exit(1)
      end if

      ! define the local domain
      domain_desc(did)%tile  = tile
      domain_desc(did)%device_id = DeviceCPU
      domain_desc(did)%cube  = cube
      domain_desc(did)%first = [(tile_coord(1)-1)*blkx+1, (tile_coord(2)-1)*blky+1]
      domain_desc(did)%last  = domain_desc(did)%first + [blkx, blky] - 1
      print *, "rank ", rank, " tile ", tile, " first ", domain_desc(did)%first, " last ", domain_desc(did)%last

      allocate(data_scalar(blkx+sum(halo(1:2)), blky+sum(halo(3:4)), cube(2)), source=-1.0)
      data_scalar(:, :, :) = rank*ntiles+did
      data_ptr(did)%ptr => data_scalar

      call ghex_field_init(field_desc, data_ptr(did)%ptr, halo)
      call ghex_domain_add_field(domain_desc(did), field_desc)

      did = did + 1
      j = j + 1
    end do
    i = i + 1
  end do
   
  call mpi_barrier(mpi_comm_world, mpi_err)
 
  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

  ! create communication object
  call ghex_co_init(co)

  ! exchange halos
  eh = ghex_exchange(co, ed)
  call ghex_wait(eh)
  call ghex_free(domain_desc(1))
  call ghex_free(ed)

  call mpi_barrier(mpi_comm_world, mpi_err)
  
  call ghex_finalize()
  call mpi_finalize(mpi_err)

END PROGRAM test_f_cubed_sphere
