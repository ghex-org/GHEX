PROGRAM test_f_cubed_sphere
  use ghex_mod

  implicit none

  include 'mpif.h'

  real    :: tic, toc
  integer :: mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size, world_rank
  integer :: tmp, i, j, f
  integer :: ntiles = 6
  integer :: tile_dims(2) = [2, 2]         ! each tile is split into tile_dims ranks, in X and Y dimensions
  integer :: tile = -1, tile_coord(2)      ! which tile do we belong to, and what are our tile coordinates
  integer :: cube(2) = [10, 6]             ! dimensions of the tile domains, (nx*nx*ndepth)
  integer :: blkx, blky
  integer :: halo(4)                       ! halo definition
  integer :: first(2), last(2)
  integer :: extents4(4)

  ! exchange scalar and vector fields
  real(ghex_fp_kind), dimension(:,:,:), pointer :: data_scalar
  real(ghex_fp_kind), dimension(:,:,:,:), pointer :: data_vector1, data_vector2
  integer :: n_components = 3

  ! GHEX stuff
  type(ghex_communicator)                      :: comm         ! communicator
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
  
  ! create ghex communicator
  comm = ghex_comm_new()
  
  ! halo width
  halo(:) = 1

  ! check if we have the right number of ranks
  if (size /= product(tile_dims)*ntiles) then
    print *, "Usage: this test must be executed with ", product(tile_dims)*ntiles, " mpi ranks"
    call ghex_finalize()
    call mpi_finalize(mpi_err)
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
    call ghex_finalize()
    call mpi_finalize(mpi_err)
    call exit(1)
  end if

  ! create communication object
  call ghex_co_init(co, comm)

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
  
  ! exchange halos
  eh = ghex_exchange(co, ed)
  call ghex_wait(eh)

  ! cleanups
  call ghex_free(domain_desc)
  call ghex_free(ed)

  ! vector field exchange: try both layouts at the same time
  call ghex_cubed_sphere_domain_init(domain_desc, tile, cube, first, last)  

  allocate(data_vector1(blkx+sum(halo(1:2)), blky+sum(halo(3:4)), cube(2), n_components), source=-1.0)
  data_vector1(:, :, :, 1) = 10*rank
  data_vector1(:, :, :, 2) = 100*rank
  data_vector1(:, :, :, 3) = 1000*rank
  call ghex_field_init(field_desc, data_vector1, halo, is_vector=.true.)
  call ghex_domain_add_field(domain_desc, field_desc)
  call ghex_free(field_desc)

  allocate(data_vector2(n_components, blkx+sum(halo(1:2)), blky+sum(halo(3:4)), cube(2)), source=-1.0)
  data_vector2(1, :, :, :) = 10*rank
  data_vector2(2, :, :, :) = 100*rank
  data_vector2(3, :, :, :) = 1000*rank
  ! TODO: this segfaults now, needs a look
  ! call ghex_field_init(field_desc, data_vector2, halo, is_vector=.true., layout=LayoutFieldFirst)
  ! call ghex_domain_add_field(domain_desc, field_desc)
  ! call ghex_free(field_desc)
  
  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

  ! exchange halos
  eh = ghex_exchange(co, ed)
  call ghex_wait(eh) 

  ! cleanups
  call ghex_free(domain_desc)
  call ghex_free(ed)

  if (rank == 1) then
    extents4 = shape(data_vector1, 4)
    do f=1,extents4(4)
      do i=1,extents4(1)
        do j=1,extents4(2)
          write (*, fmt="(f7.0)", advance="no") data_vector1(j,i,1,f)
        end do
        write(*,*)
      end do
      write(*,*)
    end do

    extents4 = shape(data_vector2, 4)
    do f=1,extents4(1)
      do i=1,extents4(2)
        do j=1,extents4(3)
          write (*, fmt="(f7.0)", advance="no") data_vector2(f,j,i,1)
        end do
        write(*,*)
      end do
      write(*,*)
    end do
  end if
  call ghex_comm_barrier(comm, GhexBarrierGlobal)

  ! cleanups
  call ghex_free(domain_desc)
  call ghex_free(ed)
  call ghex_finalize()
  call mpi_finalize(mpi_err)

END PROGRAM test_f_cubed_sphere
