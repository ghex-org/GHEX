PROGRAM test_halo_exchange
  use iso_c_binding
  use ghex_mod

  implicit none

  include 'mpif.h'

  character(len=512) :: arg
  real    :: tic, toc
  integer :: mpi_err, mpi_threading
  integer :: nthreads = 1, world_size, world_rank
  integer :: tmp, i, it
  integer :: gfirst(3), glast(3)       ! global index space
  integer :: first(3), last(3)         ! local index space

  integer :: cart_dim(3) = [0, 0, 0]
  integer :: ldim(3) = [128, 128, 128] ! dimensions of the local domains
  logical :: periodic(3) = [.true.,.true.,.true.] ! for MPI_Cart_create
  integer :: rank_coord(3)             ! local rank coordinates in a cartesian rank space
  integer :: halo(6)                   ! halo definition
  integer :: mb = 5                    ! halo width
  integer :: niters = 1000
  integer, parameter :: nfields_max = 8
  integer :: nfields

  ! -------------- variables used by the Bifrost-like implementation
  integer :: xsb, xeb, ysb, yeb, zsb, zeb
  integer :: xs , xe , ys , ye , zs , ze
  integer :: xr , xrb, yr , yrb, zr , zrb
  integer :: C_CART, R_XUP, R_XDN, R_YUP, R_YDN, R_ZUP, R_ZDN
  integer(kind=4) :: T_SENDXUP_REAL4,T_RECVXUP_REAL4,T_SENDXDN_REAL4,T_RECVXDN_REAL4
  integer(kind=4) :: T_SENDYUP_REAL4,T_RECVYUP_REAL4,T_SENDYDN_REAL4,T_RECVYDN_REAL4
  integer(kind=4) :: T_SENDZUP_REAL4,T_RECVZUP_REAL4,T_SENDZDN_REAL4,T_RECVZDN_REAL4
  integer(kind=4),dimension(MPI_STATUS_SIZE) :: status
  ! --------------

  type hptr
     real(kind=4), dimension(:,:,:), pointer :: ptr
  end type hptr

  real,dimension(:,:,:),allocatable, target :: v1, v2, v3, v4, v5, v6, v7, v8

  ! exchange 8 data cubes
  type(hptr) :: data_ptr(nfields_max)

  ! GHEX stuff
  type(ghex_communicator)                :: comm         ! communicator
  type(ghex_struct_field)                :: field_desc   ! field descriptor

  ! single domain, multiple fields
  type(ghex_struct_domain)               :: domain_desc  ! domain descriptor
  type(ghex_struct_communication_object) :: co           ! communication object
  type(ghex_struct_exchange_descriptor)  :: ed           ! exchange descriptor
  type(ghex_struct_exchange_handle)      :: eh           ! exchange handle

  ! one field per domain, multiple domains
  type(ghex_struct_domain),               dimension(:) :: domain_descs(nfields_max)
  type(ghex_struct_communication_object), dimension(:) :: cos(nfields_max)
  type(ghex_struct_exchange_descriptor),  dimension(:) :: eds(nfields_max)
  procedure(f_cart_rank_neighbor), pointer :: p_cart_nbor
  p_cart_nbor => cart_rank_neighbor
  
  ! init mpi
  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
  call mpi_comm_rank(mpi_comm_world, world_rank, mpi_err)
  call mpi_comm_size(mpi_comm_world, world_size, mpi_err)

  if (command_argument_count() < 4) then
    if (world_rank==0) then
      print *, "Usage: <benchmark> [grid size] [niters] [halo size] [num fields] <cart dims>"
    end if
    call mpi_barrier(mpi_comm_world, mpi_err)
    call mpi_finalize(mpi_err)
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

  ! number of fields
  call get_command_argument(4, arg)
  read(arg,*) nfields

  ! Cartesian communicator decomposition
  ! global dimensions
  if (command_argument_count() > 4) then
    call get_command_argument(5, arg)
    read(arg,*) cart_dim(1)
    call get_command_argument(6, arg)
    read(arg,*) cart_dim(2)
    call get_command_argument(7, arg)
    read(arg,*) cart_dim(3)
  else
    call mpi_dims_create(world_size, 3, cart_dim, mpi_err)
  end if

  if (world_rank == 0) then
    ! check if correct number of ranks
    if (product(cart_dim) /= world_size) then
      write (*,"(a, i4, a, i4, a)") "Number of ranks (", world_size, ") doesn't match the domain decomposition (", product(cart_dim), ")"
      call exit(1)
    end if
  end if

  call mpi_cart_create(mpi_comm_world, 3, cart_dim, periodic, .true., C_CART, mpi_err)

  ! ------------- common init
  ! halo information
  halo(:) = 0
  halo(1:2) = mb
  halo(3:4) = mb
  halo(5:6) = mb

  call mpi_comm_rank(C_CART, world_rank, mpi_err)
  if (world_rank==0) then
    print *, "halos: ", halo
    print *, "domain dist: ", cart_dim
    print *, "domain size: ", ldim
  end if

  ! local indices in the rank index space
  call mpi_cart_coords(C_CART, world_rank, 3, rank_coord, mpi_err)
  
  ! define the global index domain
  gfirst = [1, 1, 1]
  glast = cart_dim * ldim

  ! define the local domain
  first = (rank_coord) * ldim + 1
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

  xr  = xe  - xs + 1
  yr  = ye  - ys + 1
  zr  = ze  - zs + 1
  xrb = xeb - xsb + 1
  yrb = yeb - ysb + 1
  zrb = zeb - zsb + 1

  ! allocate and initialize data cubes
  i = 1
  do while (i <= nfields)
    allocate(data_ptr(i)%ptr(xsb:xeb, ysb:yeb, zsb:zeb), source=0.0)
    i = i+1
  end do

  if (.true.) then

    ! ------ ghex
    call ghex_init(nthreads, C_CART)

    ! create ghex communicator
    comm = ghex_comm_new()

    ! create communication object
    call ghex_co_init(co, comm)

    call ghex_domain_init(domain_desc, world_rank, first, last, gfirst, glast, p_cart_nbor)

    ! make individual copies for sequenced comm
    i = 1
    do while (i <= nfields)
      call ghex_domain_init(domain_descs(i), world_rank, first, last, gfirst, glast, p_cart_nbor)
      i = i+1
    end do

    ! ---- GHEX tests ----
    ! initialize the field datastructure
    i = 1
    do while (i <= nfields)
      call ghex_field_init(field_desc, data_ptr(i)%ptr, halo, periodic=periodic)
      call ghex_domain_add_field(domain_desc, field_desc)
      call ghex_free(field_desc)
      i = i+1
    end do

    ! compute the halo information for all domains and fields
    ed = ghex_exchange_desc_new(domain_desc)

    ! initialize data cubes
    i = 1
    do while (i <= nfields)
      data_ptr(i)%ptr(:,:,:) = -1
      data_ptr(i)%ptr(xs:xe, ys:ye, zs:ze) = world_rank+i
      i = i+1
    end do

    ! warmup
    it = 0
    do while (it < 10)
      eh = ghex_exchange(co, ed)
      call ghex_wait(eh)
      call ghex_free(eh)
      it = it+1;
    end do
    call ghex_comm_barrier(comm, GhexBarrierGlobal)
    call test_exchange(data_ptr, rank_coord)

    ! time loop
    it = 0
    call cpu_time(tic)
    do while (it < niters)
      eh = ghex_exchange(co, ed)
      call ghex_wait(eh)
      call ghex_free(eh)
      it = it+1
    end do
    call ghex_comm_barrier(comm, GhexBarrierGlobal)
    call cpu_time(toc)
    if (world_rank == 0) then
      print *, "exchange GHEX:      ", (toc-tic)
    end if

    call ghex_free(co)
    call ghex_free(domain_desc)
    call ghex_free(ed)

    ! sequenced tests
    if(.true.) then

      ! initialize the field datastructure
      ! compute the halo information for all domains and fields
      i = 1
      do while (i <= nfields)
        call ghex_field_init(field_desc, data_ptr(i)%ptr, halo, periodic=periodic)
        call ghex_domain_add_field(domain_descs(i), field_desc)
        call ghex_free(field_desc)
        eds(i) = ghex_exchange_desc_new(domain_descs(i))
        i = i+1
      end do

      ! create communication objects
      i = 1
      do while (i <= nfields)
        call ghex_co_init(cos(i), comm)
        i = i+1
      end do

      ! initialize data cubes
      i = 1
      do while (i <= nfields)
        data_ptr(i)%ptr(:,:,:) = -1
        data_ptr(i)%ptr(xs:xe, ys:ye, zs:ze) = world_rank+i
        i = i+1
      end do

      ! exchange halos
      it = 0
      do while (it < 10)
        i = 1
        do while (i <= nfields)
          eh = ghex_exchange(cos(i), eds(i)); call ghex_wait(eh)
          i = i+1
        end do
        it = it+1
      end do
      call ghex_comm_barrier(comm, GhexBarrierGlobal)
      call test_exchange(data_ptr, rank_coord)

      it = 0
      call cpu_time(tic)
      do while (it < niters)
        i = 1
        do while (i <= nfields)
          eh = ghex_exchange(cos(i), eds(i)); call ghex_wait(eh)
          i = i+1
        end do
        it = it+1
      end do
      call ghex_comm_barrier(comm, GhexBarrierGlobal)
      call cpu_time(toc)
      if (world_rank == 0) then
        print *, "exchange sequenced (multiple COs):      ", (toc-tic);
      end if

      ! WARNING: the below cannot be run for bulk communicatio objects
      if(.false.) then
        call ghex_co_init(co, comm)

        ! initialize data cubes
        i = 1
        do while (i <= nfields)
          data_ptr(i)%ptr(:,:,:) = -1
          data_ptr(i)%ptr(xs:xe, ys:ye, zs:ze) = world_rank+i
          i = i+1
        end do

        ! exchange halos
        it = 0
        do while (it < 10)
          i = 1
          do while (i <= nfields)
            eh = ghex_exchange(co, eds(i)); call ghex_wait(eh)
            i = i+1
          end do
          it = it+1
        end do
        call ghex_comm_barrier(comm, GhexBarrierGlobal)
        call test_exchange(data_ptr, rank_coord)

        it = 0
        call cpu_time(tic)
        do while (it < niters)
          i = 1
          do while (i <= nfields)
            eh = ghex_exchange(co, eds(i)); call ghex_wait(eh)
            i = i+1
          end do
          it = it+1
        end do
        call ghex_comm_barrier(comm, GhexBarrierGlobal)
        call cpu_time(toc)
        if (world_rank == 0) then
          print *, "exchange sequenced (single CO):      ", (toc-tic);
        end if
      end if

      i = 1
      do while (i <= nfields)
        call ghex_free(domain_descs(i))
        call ghex_free(cos(i))
        call ghex_free(eds(i))
        i = i+1
      end do
    end if

    ! cleanup GHEX
    call ghex_comm_barrier(comm, GhexBarrierGlobal)
    call ghex_finalize()

  end if

  ! ---- BIFROST-like comm ----
  if (.true.) then

    ! compute neighbor information
    call init_mpi_nbors(rank_coord)
    call exchange_subarray_init


    ! MPI EXCHANGE (1)
    ! initialize data cubes
    i = 1
    do while (i <= nfields)
      data_ptr(i)%ptr(:,:,:) = -1
      data_ptr(i)%ptr(xs:xe, ys:ye, zs:ze) = world_rank+i
      i = i+1
    end do

    ! warmup
    i = 1
    do while (i <= nfields)
      call exchange_subarray(data_ptr(i)%ptr)
      i = i+1
    end do
    call mpi_barrier(mpi_comm_world, mpi_err)
    call test_exchange(data_ptr, rank_coord)

    ! time loop
    call cpu_time(tic)
    it = 0
    do while (it < niters)
      i = 1
      do while (i <= nfields)
        call exchange_subarray(data_ptr(i)%ptr)
        i = i+1
      end do
      it = it+1
    end do
    call mpi_barrier(mpi_comm_world, mpi_err)
    call cpu_time(toc)
    if (world_rank == 0) then
      print *, "subarray exchange:      ", (toc-tic)
    end if


    ! MPI EXCHANGE (2)
    ! initialize data cubes
    i = 1
    do while (i <= nfields)
      data_ptr(i)%ptr(:,:,:) = -1
      data_ptr(i)%ptr(xs:xe, ys:ye, zs:ze) = world_rank+i
      i = i+1
    end do

    ! warmup
    i = 1
    do while (i <= nfields)
      call update_sendrecv(data_ptr(i)%ptr)
      i = i+1
    end do
    call mpi_barrier(mpi_comm_world, mpi_err)
    call test_exchange(data_ptr, rank_coord)

    ! time loop
    it = 0
    call cpu_time(tic)
    do while (it < niters)
      i = 1
      do while (i <= nfields)
        call update_sendrecv(data_ptr(i)%ptr)
        i = i+1
      end do
      it = it+1
    end do
    call mpi_barrier(mpi_comm_world, mpi_err)
    call cpu_time(toc)
    if (world_rank == 0) then
      print *, "bifrost exchange 1:      ", (toc-tic)
    end if


    ! MPI EXCHANGE (3)
    ! initialize data cubes
    i = 1
    do while (i <= nfields)
      data_ptr(i)%ptr(:,:,:) = -1
      data_ptr(i)%ptr(xs:xe, ys:ye, zs:ze) = world_rank+i
      i = i+1
    end do

    ! warmup
    i = 1
    do while (i <= nfields)
      call update_sendrecv_2(data_ptr(i)%ptr)
      i = i+1
    end do
    call mpi_barrier(mpi_comm_world, mpi_err)
    call test_exchange(data_ptr, rank_coord)

    ! time loop
    call cpu_time(tic)
    it = 0
    do while (it < niters)
      i = 1
      do while (i <= nfields)
        call update_sendrecv_2(data_ptr(i)%ptr)
        i = i+1
      end do
      it = it+1
    end do
    call mpi_barrier(mpi_comm_world, mpi_err)
    call cpu_time(toc)
    if (world_rank == 0) then
      print *, "bifrost exchange 2:      ", (toc-tic)
    end if
  end if

  call mpi_barrier(mpi_comm_world, mpi_err)
  call mpi_finalize(mpi_err)

contains

  function get_nbor(icoord, shift, idx)
    integer, intent(in) :: icoord(3)
    integer :: shift, idx
    integer :: get_nbor
    integer :: coord(3)

    coord = icoord
    coord(idx) = coord(idx)+shift
    call mpi_cart_rank(C_CART, coord, get_nbor, mpi_err)
  end function get_nbor

  subroutine init_mpi_nbors(rank_coord)
    integer :: rank_coord(3)

    ! all dimensions are periodic, also for fully local domains
    R_XUP = get_nbor(rank_coord, +1, 1);
    R_XDN = get_nbor(rank_coord, -1, 1);

    R_YUP = get_nbor(rank_coord, +1, 2);
    R_YDN = get_nbor(rank_coord, -1, 2);

    R_ZUP = get_nbor(rank_coord, +1, 3);
    R_ZDN = get_nbor(rank_coord, -1, 3);
  end subroutine init_mpi_nbors


  ! -------------------------------------------------------------
  ! Bifrost-like communication with 3 synchroneous steps: sendrecv on array parts
  ! -------------------------------------------------------------
  subroutine update_sendrecv(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call comm_x(f)
    call comm_y(f)
    call comm_z(f)
  end subroutine update_sendrecv

  subroutine comm_x(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call MPI_SENDRECV(f(xe-(mb-1) :xe         , &
      ys        :ye         , &
      zs        :ze        ), &
      mb*yr*zr,MPI_REAL4,R_XUP,1  , &
      f(xsb     :xsb+(mb-1) , &
      ys        :ye         , &
      zs        :ze        ), &
      mb*yr*zr,MPI_REAL4,R_XDN,1  , &
      C_CART,status,mpi_err)

    call MPI_SENDRECV(f(xs        :xs+(mb-1)  , &
      ys        :ye         , &
      zs        :ze        ), &
      mb*yr*zr,MPI_REAL4,R_XDN,2  , &
      f(xeb-(mb-1):xeb        , &
      ys        :ye         , &
      zs        :ze        ), &
      mb*yr*zr,MPI_REAL4,R_XUP,2  , &
      C_CART,status,mpi_err)

  end subroutine comm_x

  subroutine comm_y(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call MPI_SENDRECV(f(xsb       :xeb              , &
      ye-(mb-1):ye              , &
      zs       :ze       )      , &
      xrb*mb*zr,MPI_REAL4,R_YUP,3, &
      f(xsb       :xeb              , &
      ysb         :ysb+(mb-1)         , &
      zs       :ze       )      , &
      xrb*mb*zr,MPI_REAL4,R_YDN,3, &
      C_CART,status,mpi_err)

    call MPI_SENDRECV(f(xsb       :xeb              , &
      ys       :ys+(mb-1)       , &
      zs       :ze       )      , &
      xrb*mb*zr,MPI_REAL4,R_YDN,4          , &
      f(xsb       :xeb              , &
      yeb-(mb-1)  :yeb                , &
      zs       :ze       )      , &
      xrb*mb*zr,MPI_REAL4,R_YUP,4           , &
      C_CART,status,mpi_err)

  end subroutine comm_y

  subroutine comm_z(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call MPI_SENDRECV( &
      f(xsb        :xeb              , &
      ysb        :yeb              , &
      ze-(mb-1) :ze          )   , &
      xrb*yrb*mb,MPI_REAL4,R_ZUP,5, &
      f(xsb        :xeb              , &
      ysb        :yeb              , &
      zsb       :zsb+(mb-1)  )   , &
      xrb*yrb*mb,MPI_REAL4,R_ZDN,5, &
      C_CART,status,mpi_err)

    call MPI_SENDRECV( &
      f(xsb        :xeb              , &
      ysb        :yeb              , &
      zs        :zs+(mb-1))      , &
      xrb*yrb*mb,MPI_REAL4,R_ZDN,6, &
      f(xsb        :xeb              , &
      ysb        :yeb              , &
      zeb-(mb-1):zeb         )   , &
      xrb*yrb*mb,MPI_REAL4,R_ZUP,6, &
      C_CART,status,mpi_err)
  end subroutine comm_z


  ! a version with explicit copy of halo areas into a buffer
  subroutine update_sendrecv_2(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call comm_x_2(f)
    call comm_y_2(f)
    call comm_z_2(f)
  end subroutine update_sendrecv_2

  subroutine comm_x_2(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f
    real(kind=4),dimension(mb*yr*zr) :: sbuff, rbuff

    sbuff(:) = reshape(f(xe-(mb-1) :xe , &
      ys        :ye         , &
      zs        :ze         ), (/mb*yr*zr/))
    call MPI_SENDRECV( sbuff,         &
      mb*yr*zr,MPI_REAL4,R_XUP,1  , &
      rbuff,                       &
      mb*yr*zr,MPI_REAL4,R_XDN,1  , &
      C_CART,status,mpi_err)
    f(xsb     :xsb+(mb-1) , &
      ys        :ye         , &
      zs        :ze        ) = reshape(rbuff, (/mb, yr, zr/));

    sbuff(:) = reshape(f(xs        :xs+(mb-1)  , &
      ys        :ye         , &
      zs        :ze         ), (/mb*yr*zr/))
    call MPI_SENDRECV(sbuff,          &
      mb*yr*zr,MPI_REAL4,R_XDN,2  , &
      rbuff,                       &
      mb*yr*zr,MPI_REAL4,R_XUP,2  , &
      C_CART,status,mpi_err)
    f(xeb-(mb-1):xeb        , &
      ys        :ye         , &
      zs        :ze        ) = reshape(rbuff, (/mb, yr, zr/));

  end subroutine comm_x_2

  subroutine comm_y_2(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f
    real(kind=4),dimension(xrb*mb*zr) :: sbuff, rbuff

    sbuff = reshape(f(xsb       :xeb              , &
      ye-(mb-1):ye              , &
      zs       :ze       )      , &
      (/xrb*mb*zr/));
    call MPI_SENDRECV( sbuff,        &
      xrb*mb*zr,MPI_REAL4,R_YUP,3, &
      rbuff,                      &
      xrb*mb*zr,MPI_REAL4,R_YDN,3, &
      C_CART,status,mpi_err)
    f(xsb       :xeb          , &
      ysb         :ysb+(mb-1)   , &
      zs       :ze       ) = reshape(rbuff, (/xrb, mb, zr/));

    sbuff = reshape(f(xsb       :xeb              , &
      ys       :ys+(mb-1)       , &
      zs       :ze       )      , &
      (/xrb*mb*zr/));
    call MPI_SENDRECV(sbuff,         &
      xrb*mb*zr,MPI_REAL4,R_YDN,4, &
      rbuff,                      &
      xrb*mb*zr,MPI_REAL4,R_YUP,4, &
      C_CART,status,mpi_err)
    f(xsb       :xeb          , &
      yeb-(mb-1)  :yeb          , &
      zs       :ze       ) = reshape(rbuff, (/xrb, mb, zr/));
  end subroutine comm_y_2

  subroutine comm_z_2(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f
    real(kind=4),dimension(xrb*yrb*mb) :: sbuff, rbuff

    sbuff = reshape(f(xsb        :xeb              , &
      ysb        :yeb              , &
      ze-(mb-1) :ze          )   , &
      (/xrb*yrb*mb/))
    call MPI_SENDRECV( sbuff, &
      xrb*yrb*mb,MPI_REAL4,R_ZUP,5, &
      rbuff, &
      xrb*yrb*mb,MPI_REAL4,R_ZDN,5, &
      C_CART,status,mpi_err)
    f(xsb        :xeb              , &
      ysb        :yeb              , &
      zsb       :zsb+(mb-1)  ) = reshape(rbuff, (/xrb,yrb,mb/))

    sbuff = reshape(f(xsb        :xeb              , &
      ysb        :yeb              , &
      zs        :zs+(mb-1))      , &
      (/xrb*yrb*mb/));
    call MPI_SENDRECV( sbuff, &
      xrb*yrb*mb,MPI_REAL4,R_ZDN,6, &
      rbuff, &
      xrb*yrb*mb,MPI_REAL4,R_ZUP,6, &
      C_CART,status,mpi_err)
    f(xsb        :xeb              , &
      ysb        :yeb              , &
      zeb-(mb-1):zeb         ) = reshape(rbuff, (/xrb,yrb,mb/))
  end subroutine comm_z_2


  ! a version with subarrays
  subroutine  exchange_subarray_init()
    integer,dimension(3) :: sizeb,sizel,sizes

    sizeb=(/xrb,yrb,zrb/)

    ! --- X halos
    sizel=(/mb,yrb,zrb/)

    ! send up
    sizes=(/xrb-2*mb,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDXUP_REAL4,mpi_err)
    call mpi_type_commit(T_SENDXUP_REAL4,mpi_err)

    ! recv up
    sizes=(/xrb-mb,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVXUP_REAL4,mpi_err)
    call mpi_type_commit(T_RECVXUP_REAL4,mpi_err)

    ! send down
    sizes=(/mb,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDXDN_REAL4,mpi_err)
    call mpi_type_commit(T_SENDXDN_REAL4,mpi_err)

    ! recv down
    sizes=(/0,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVXDN_REAL4,mpi_err)
    call mpi_type_commit(T_RECVXDN_REAL4,mpi_err)

    ! --- Y halos
    sizel=(/xrb,mb,zrb/)

    ! send up
    sizes=(/0,yrb-2*mb,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDYUP_REAL4,mpi_err)
    call mpi_type_commit(T_SENDYUP_REAL4,mpi_err)

    ! recv up
    sizes=(/0,yrb-mb,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVYUP_REAL4,mpi_err)
    call mpi_type_commit(T_RECVYUP_REAL4,mpi_err)

    ! send down
    sizes=(/0,mb,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDYDN_REAL4,mpi_err)
    call mpi_type_commit(T_SENDYDN_REAL4,mpi_err)

    ! recv down
    sizes=(/0,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVYDN_REAL4,mpi_err)
    call mpi_type_commit(T_RECVYDN_REAL4,mpi_err)

    ! --- Z halos
    sizel=(/xrb,yrb,mb/)

    ! send up
    sizes=(/0,0,zrb-2*mb/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDZUP_REAL4,mpi_err)
    call mpi_type_commit(T_SENDZUP_REAL4,mpi_err)

    ! recv up
    sizes=(/0,0,zrb-mb/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVZUP_REAL4,mpi_err)
    call mpi_type_commit(T_RECVZUP_REAL4,mpi_err)

    ! send down
    sizes=(/0,0,mb/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDZDN_REAL4,mpi_err)
    call mpi_type_commit(T_SENDZDN_REAL4,mpi_err)

    ! recv down
    sizes=(/0,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVZDN_REAL4,mpi_err)
    call mpi_type_commit(T_RECVZDN_REAL4,mpi_err)
  end subroutine exchange_subarray_init

  subroutine exchange_subarray(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call comm_x_subarray(f)
    call comm_y_subarray(f)
    call comm_z_subarray(f)
  end subroutine exchange_subarray

  subroutine comm_x_subarray(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call MPI_SENDRECV(f,                &
      1, T_SENDXUP_REAL4, R_XUP, 1 , &
      f,                             &
      1, T_RECVXDN_REAL4, R_XDN, 1 , &
      C_CART,status,mpi_err)

    call MPI_SENDRECV(f,                &
      1, T_SENDXDN_REAL4, R_XDN, 1 , &
      f,                             &
      1, T_RECVXUP_REAL4, R_XUP, 1 , &
      C_CART,status,mpi_err)
  end subroutine comm_x_subarray

  subroutine comm_y_subarray(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call MPI_SENDRECV(f,                &
      1, T_SENDYUP_REAL4, R_YUP, 1 , &
      f,                             &
      1, T_RECVYDN_REAL4, R_YDN, 1 , &
      C_CART,status,mpi_err)

    call MPI_SENDRECV(f,                &
      1, T_SENDYDN_REAL4, R_YDN, 1 , &
      f,                             &
      1, T_RECVYUP_REAL4, R_YUP, 1 , &
      C_CART,status,mpi_err)
  end subroutine comm_y_subarray

  subroutine comm_z_subarray(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call MPI_SENDRECV(f,                &
      1, T_SENDZUP_REAL4, R_ZUP, 1 , &
      f,                             &
      1, T_RECVZDN_REAL4, R_ZDN, 1 , &
      C_CART,status,mpi_err)

    call MPI_SENDRECV(f,                &
      1, T_SENDZDN_REAL4, R_ZDN, 1 , &
      f,                             &
      1, T_RECVZUP_REAL4, R_ZUP, 1 , &
      C_CART,status,mpi_err)
  end subroutine comm_z_subarray

  subroutine test_exchange(data_ptr, rank_coord)
    type(hptr), intent(in), dimension(:) :: data_ptr
    integer, intent(in) :: rank_coord(3)             ! local rank coordinates in a cartesian rank space    
    integer :: nbor_coord(3), nbor_rank, ix, iy, iz, jx, jy, jz
    real(kind=4), dimension(:,:,:), pointer :: data
    integer :: isx(-1:1), iex(-1:1)
    integer :: isy(-1:1), iey(-1:1)
    integer :: isz(-1:1), iez(-1:1)
    integer :: i, j, k, did
    logical :: err

    err = .false.
    isx = (/xsb, xs, xe+1/);
    iex = (/xs-1, xe, xeb/);

    isy = (/ysb, ys, ye+1/);
    iey = (/ys-1, ye, yeb/);

    isz = (/zsb, zs, ze+1/);
    iez = (/zs-1, ze, zeb/);

    did = 1
    do while(did<=nfields)
      i = -1
      do while (i<=1)
        j = -1
        do while (j<=1)
          k = -1
          do while (k<=1)

            ! get nbor rank
            nbor_coord = rank_coord + (/i,j,k/);
            call mpi_cart_rank(C_CART, nbor_coord, nbor_rank, mpi_err)

            ! check cube values
            if (.not.all(data_ptr(did)%ptr(isx(i):iex(i), isy(j):iey(j), isz(k):iez(k))==nbor_rank+did)) then
              print *, "wrong halo ", i, j, k
              err = .true.
            end if
            k = k+1
          end do
          j = j+1
        end do
        i = i+1
      end do
      did = did+1
    end do
    if (err) then
      call exit(1)
    end if
  end subroutine test_exchange

  subroutine print_cube_3d(data)
    real(ghex_fp_kind), dimension(:,:,:), pointer :: data
    integer :: extents(3), i, j, k

    do i=xsb,xeb
      do j=ysb,yeb
        do k=zsb,zeb
          write (*, fmt="(f3.0)", advance="no") data(j,i,k)
        end do
        write(*,*)
      end do
      write(*,*)
    end do
  end subroutine print_cube_3d

  subroutine cart_rank_neighbor (id, offset_x, offset_y, offset_z, nbid_out, nbrank_out) bind(c)
    use iso_c_binding
    integer(c_int), value, intent(in) :: id, offset_x, offset_y, offset_z
    integer(c_int), intent(out) :: nbid_out, nbrank_out
    integer :: coord(3)

    call mpi_cart_coords(C_CART, id, 3, coord, mpi_err);
    coord(1) = coord(1) + offset_x;
    coord(2) = coord(2) + offset_y;
    coord(3) = coord(3) + offset_z;
    call mpi_cart_rank(C_CART, coord, nbrank_out, mpi_err);
    nbid_out = nbrank_out;
  end subroutine cart_rank_neighbor

END PROGRAM test_halo_exchange
