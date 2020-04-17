PROGRAM test_f_cubed_sphere
  use omp_lib
  use ghex_mod

  implicit none  

  include 'mpif.h'  

  real    :: tic, toc
  integer :: ierr, mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size, world_rank
  integer :: tmp, i
  integer :: ntiles = 6
  integer :: tile_dims(2) = [2, 2]         ! each tile is split into tile_dims ranks, in X and Y dimensions
  integer :: tile = -1, tile_coord(2)      ! which tile do we belong to, and what are our tile coordinates
  integer :: cube(2) = [64, 20]            ! dimensions of the tile domains, (nx*nx*ndepth)
  integer :: blkx, blky
  integer :: halo(4), mb                   ! halo definition
  integer :: niters = 100
  
  ! exchange 8 data cubes
  real(ghex_fp_kind), dimension(:,:,:), pointer :: data1, data2, data3, data4
  real(ghex_fp_kind), dimension(:,:,:), pointer :: data5, data6, data7, data8

  ! GHEX stuff
  type(ghex_cubed_sphere_domain), target, dimension(:) :: domain_desc(1), d1(1), d2(1), d3(1), d4(1), d5(1), d6(1), d7(1), d8(1)
  type(ghex_cubed_sphere_field)     :: field_desc
  type(ghex_cubed_sphere_communication_object) :: co, co1, co2, co3, co4, co5, co6, co7, co8
  type(ghex_cubed_sphere_exchange_descriptor)  :: ed, ed1, ed2, ed3, ed4, ed5, ed6, ed7, ed8
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
  if (size /= product(tile_dims)*ntiles) then
    print *, "Usage: this test must be executed with ", product(tile_dims)*ntiles, " mpi ranks"
    call exit(1)
  end if

  ! rank to tile
  tile = modulo(rank, ntiles)+1
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
  domain_desc(1)%tile  = tile
  domain_desc(1)%device_id = DeviceCPU
  domain_desc(1)%cube  = cube
  domain_desc(1)%first = [(tile_coord(1)-1)*blkx+1, (tile_coord(2)-1)*blky+1]
  domain_desc(1)%last  = domain_desc(1)%first + [blkx, blky] - 1
  ! print *, "rank ", rank, " tile ", tile, " first ", domain_desc(1)%first, " last ", domain_desc(1)%last

  allocate(data1(blkx+sum(halo(1:2)), blky+sum(halo(3:4)), cube(2)), source=-1.0)
  data1(:, :, :) = rank  

  call ghex_field_init(field_desc, data1, halo)
  call ghex_domain_add_field(domain_desc(1), field_desc)

  ! compute the halo information for all domains and fields
  ! ed = ghex_exchange_desc_new(domain_desc)  
  print *, "rank ", rank

  call mpi_barrier(mpi_comm_world, mpi_err)
  call ghex_delete(ed)
  call ghex_finalize()
  call mpi_finalize(mpi_err)

END PROGRAM test_f_cubed_sphere
