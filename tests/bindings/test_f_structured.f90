PROGRAM test_halo_exchange
  use omp_lib
  use ghex_mod
  use hwcart_mod

  implicit none

  character(len=512) :: arg
  real    :: tic, toc
  integer :: ierr, mpi_err, mpi_threading
  integer :: nthreads = 1, world_size, world_rank
  integer :: tmp, i, it
  integer :: gfirst(3), glast(3)       ! global index space
  integer :: first(3), last(3)         ! local index space

  ! hierarchical decomposition
  integer :: domain(5) = 0
  integer :: topology(3,5) = 1
  integer :: level_rank(5) = -1
  integer :: cart_dim(3) = 1
  character(len=3) :: cart_order_arg
  integer :: cart_order = HWCartOrderZYX

  logical :: remap = .false.           ! remap MPI ranks
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

  ! HWCART stuff
  type(hwcart_topo_t) :: hwcart_topo

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

  ! init mpi
  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
  call mpi_comm_rank(mpi_comm_world, world_rank, mpi_err)
  call mpi_comm_size(mpi_comm_world, world_size, mpi_err)

  if (command_argument_count() < 4) then
    if (world_rank==0) then
      print *, "Usage: <benchmark> [grid size] [niters] [halo size] [num fields] <l3 dims :3> <numa dims :3> <socket dims :3> <node dims :3> <global dims :3> <cart_order>"
    end if
    call mpi_barrier(mpi_comm_world, mpi_err)
    call mpi_finalize(mpi_err)
    call exit(1)
  end if

  ! init HWCART
  ierr = hwcart_init(hwcart_topo)
  if (ierr /= 0) then
    call MPI_Finalize(ierr);
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

  ! hierarchical decomposition:
  ! 1. L3 cache block (lowest level, actual ranks)
  ! 2. numa blocks (composition of L3 blocks)
  ! 3. socket blocks (composition of numa blocks)
  ! 4. node blocks (...)
  ! 5. global grid composed of node blocks  

  ! global dimensions
  if (command_argument_count() > 4) then
    call get_command_argument(5, arg)
    read(arg,*) topology(1,5)
    call get_command_argument(6, arg)
    read(arg,*) topology(2,5)
    call get_command_argument(7, arg)
    read(arg,*) topology(3,5)
  end if

  ! rank L3 block dimensions
  if (command_argument_count() > 7) then
    remap = .true.
    call get_command_argument(8, arg)
    read(arg,*) topology(1,1)
    call get_command_argument(9, arg)
    read(arg,*) topology(2,1)
    call get_command_argument(10, arg)
    read(arg,*) topology(3,1)
  end if

  ! numa node block dimensions
  if (command_argument_count() > 10) then
    call get_command_argument(11, arg)
    read(arg,*) topology(1,2)
    call get_command_argument(12, arg)
    read(arg,*) topology(2,2)
    call get_command_argument(13, arg)
    read(arg,*) topology(3,2)
  end if

  ! socket block dimensions
  if (command_argument_count() > 13) then
    call get_command_argument(14, arg)
    read(arg,*) topology(1,3)
    call get_command_argument(15, arg)
    read(arg,*) topology(2,3)
    call get_command_argument(16, arg)
    read(arg,*) topology(3,3)
  end if

  ! compute node block dimensions
  if (command_argument_count() > 16) then
    call get_command_argument(17, arg)
    read(arg,*) topology(1,4)
    call get_command_argument(18, arg)
    read(arg,*) topology(2,4)
    call get_command_argument(19, arg)
    read(arg,*) topology(3,4)
  end if

  if (command_argument_count() > 19) then
    call get_command_argument(20, arg)
    read(arg,*) cart_order_arg

    select case (cart_order_arg)
    case ("XYZ")
      cart_order = HWCartOrderXYZ
    case ("XZY")
      cart_order = HWCartOrderXZY
    case ("ZYX")
      cart_order = HWCartOrderZYX
    case ("YZX")
      cart_order = HWCartOrderYZX
    case ("ZXY")
      cart_order = HWCartOrderZXY
    case ("YXZ")
      cart_order = HWCartOrderYXZ
    case default
      print *, "unknown value of argument 'cart_order': ", cart_order
      call exit
    end select   
  end if

  ! global cartesian rank space dimensions
  cart_dim = product(topology, 2)

  if (world_rank == 0) then
    write (*,*) "   global block:", topology(:,5)
    if (remap) then
      write (*,*) "       L3 block:", topology(:,1)
      write (*,*) "NUMA node block:", topology(:,2)
      write (*,*) "   socket block:", topology(:,3)
      write (*,*) " shm node block:", topology(:,4)
    end if

    ! check if correct number of ranks
    if (product(cart_dim) /= world_size) then
      write (*,"(a, i4, a, i4, a)") "Number of ranks (", world_size, ") doesn't match the domain decomposition (", product(cart_dim), ")"
      call exit(1)
    end if
  end if

  if (world_rank == 0) then
    print *, "--------------------------"
  end if
  if (remap) then

    ! construct topology info to reorder the ranks
    domain(1) = HWCART_MD_CORE
    domain(2) = HWCART_MD_L3CACHE
    domain(3) = HWCART_MD_NUMA
    domain(4) = HWCART_MD_SOCKET
    domain(5) = HWCART_MD_NODE

    ierr = hwcart_create(hwcart_topo, MPI_COMM_WORLD, domain, topology, cart_order, C_CART)
    ierr = hwcart_print_rank_topology(hwcart_topo, C_CART, domain, topology, cart_order);

  else
    if (world_rank==0) then
      print *, "Using standard MPI cartesian communicator"
    end if
    call mpi_dims_create(world_size, 3, cart_dim, mpi_err)
    call mpi_cart_create(mpi_comm_world, 3, cart_dim, periodic, .true., C_CART, ierr)
    block
      integer(4) :: domain(2) = 0, topology(3,2) = 1
      domain(1) = HWCART_MD_CORE
      domain(2) = HWCART_MD_NODE
      topology(:,1) = cart_dim
      ierr = hwcart_print_rank_topology(hwcart_topo, C_CART, domain, topology, cart_order);
    end block
  end if

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
  ierr = hwcart_rank2coord(C_CART, cart_dim, world_rank, cart_order, rank_coord)

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

    call ghex_domain_init(domain_desc, world_rank, first, last, gfirst, glast, C_CART, cart_order, cart_dim)

    ! make individual copies for sequenced comm
    i = 1
    do while (i <= nfields)
      call ghex_domain_init(domain_descs(i), world_rank, first, last, gfirst, glast, C_CART, cart_order, cart_dim)
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
    if (C_CART == mpi_comm_world) then
      call mpi_cart_rank(C_CART, coord, get_nbor, ierr)
    else
      ierr = hwcart_coord2rank(C_CART, cart_dim, periodic, coord, cart_order, get_nbor)
    end if
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
      C_CART,status,ierr)

    call MPI_SENDRECV(f(xs        :xs+(mb-1)  , &
      ys        :ye         , &
      zs        :ze        ), &
      mb*yr*zr,MPI_REAL4,R_XDN,2  , &
      f(xeb-(mb-1):xeb        , &
      ys        :ye         , &
      zs        :ze        ), &
      mb*yr*zr,MPI_REAL4,R_XUP,2  , &
      C_CART,status,ierr)

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
      C_CART,status,ierr)

    call MPI_SENDRECV(f(xsb       :xeb              , &
      ys       :ys+(mb-1)       , &
      zs       :ze       )      , &
      xrb*mb*zr,MPI_REAL4,R_YDN,4          , &
      f(xsb       :xeb              , &
      yeb-(mb-1)  :yeb                , &
      zs       :ze       )      , &
      xrb*mb*zr,MPI_REAL4,R_YUP,4           , &
      C_CART,status,ierr)

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
      C_CART,status,ierr)

    call MPI_SENDRECV( &
      f(xsb        :xeb              , &
      ysb        :yeb              , &
      zs        :zs+(mb-1))      , &
      xrb*yrb*mb,MPI_REAL4,R_ZDN,6, &
      f(xsb        :xeb              , &
      ysb        :yeb              , &
      zeb-(mb-1):zeb         )   , &
      xrb*yrb*mb,MPI_REAL4,R_ZUP,6, &
      C_CART,status,ierr)
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
      C_CART,status,ierr)
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
      C_CART,status,ierr)
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
      C_CART,status,ierr)
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
      C_CART,status,ierr)
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
      C_CART,status,ierr)
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
      C_CART,status,ierr)
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
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDXUP_REAL4,ierr)
    call mpi_type_commit(T_SENDXUP_REAL4,ierr)

    ! recv up
    sizes=(/xrb-mb,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVXUP_REAL4,ierr)
    call mpi_type_commit(T_RECVXUP_REAL4,ierr)

    ! send down
    sizes=(/mb,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDXDN_REAL4,ierr)
    call mpi_type_commit(T_SENDXDN_REAL4,ierr)

    ! recv down
    sizes=(/0,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVXDN_REAL4,ierr)
    call mpi_type_commit(T_RECVXDN_REAL4,ierr)

    ! --- Y halos
    sizel=(/xrb,mb,zrb/)

    ! send up
    sizes=(/0,yrb-2*mb,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDYUP_REAL4,ierr)
    call mpi_type_commit(T_SENDYUP_REAL4,ierr)

    ! recv up
    sizes=(/0,yrb-mb,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVYUP_REAL4,ierr)
    call mpi_type_commit(T_RECVYUP_REAL4,ierr)

    ! send down
    sizes=(/0,mb,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDYDN_REAL4,ierr)
    call mpi_type_commit(T_SENDYDN_REAL4,ierr)

    ! recv down
    sizes=(/0,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVYDN_REAL4,ierr)
    call mpi_type_commit(T_RECVYDN_REAL4,ierr)

    ! --- Z halos
    sizel=(/xrb,yrb,mb/)

    ! send up
    sizes=(/0,0,zrb-2*mb/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDZUP_REAL4,ierr)
    call mpi_type_commit(T_SENDZUP_REAL4,ierr)

    ! recv up
    sizes=(/0,0,zrb-mb/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVZUP_REAL4,ierr)
    call mpi_type_commit(T_RECVZUP_REAL4,ierr)

    ! send down
    sizes=(/0,0,mb/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_SENDZDN_REAL4,ierr)
    call mpi_type_commit(T_SENDZDN_REAL4,ierr)

    ! recv down
    sizes=(/0,0,0/)
    call mpi_type_create_subarray(3,sizeb,sizel,sizes,MPI_ORDER_FORTRAN,MPI_REAL4,T_RECVZDN_REAL4,ierr)
    call mpi_type_commit(T_RECVZDN_REAL4,ierr)
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
      C_CART,status,ierr)

    call MPI_SENDRECV(f,                &
      1, T_SENDXDN_REAL4, R_XDN, 1 , &
      f,                             &
      1, T_RECVXUP_REAL4, R_XUP, 1 , &
      C_CART,status,ierr)
  end subroutine comm_x_subarray

  subroutine comm_y_subarray(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call MPI_SENDRECV(f,                &
      1, T_SENDYUP_REAL4, R_YUP, 1 , &
      f,                             &
      1, T_RECVYDN_REAL4, R_YDN, 1 , &
      C_CART,status,ierr)

    call MPI_SENDRECV(f,                &
      1, T_SENDYDN_REAL4, R_YDN, 1 , &
      f,                             &
      1, T_RECVYUP_REAL4, R_YUP, 1 , &
      C_CART,status,ierr)
  end subroutine comm_y_subarray

  subroutine comm_z_subarray(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    call MPI_SENDRECV(f,                &
      1, T_SENDZUP_REAL4, R_ZUP, 1 , &
      f,                             &
      1, T_RECVZDN_REAL4, R_ZDN, 1 , &
      C_CART,status,ierr)

    call MPI_SENDRECV(f,                &
      1, T_SENDZDN_REAL4, R_ZDN, 1 , &
      f,                             &
      1, T_RECVZUP_REAL4, R_ZUP, 1 , &
      C_CART,status,ierr)
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
            ierr = hwcart_coord2rank(C_CART, cart_dim, periodic, nbor_coord, cart_order, nbor_rank)

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
END PROGRAM test_halo_exchange
