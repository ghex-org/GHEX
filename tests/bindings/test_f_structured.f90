PROGRAM test_halo_exchange
  use omp_lib
  use ghex_mod

  implicit none  

  include 'mpif.h'  

  character(len=512) :: arg
  real    :: tic, toc
  integer :: ierr, mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size, world_rank
  integer :: tmp, i, it
  integer :: gfirst(3), glast(3)       ! global index space
  integer :: first(3), last(3)
  integer :: gdim(3) = [1, 1, 1]       ! number of domains
  integer :: ldim(3) = [128, 128, 128] ! dimensions of the local domains
  integer :: rank_coord(3)             ! local rank coordinates in a cartesian rank space
  integer :: halo(6)                   ! halo definition
  integer :: mb = 5                    ! halo width
  integer :: niters = 1000
  integer, parameter :: nfields = 8

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
  
  ! ! exchange 8 data cubes
  type(hptr) :: data_ptr(nfields)

  ! GHEX stuff
  type(ghex_struct_field)                :: field_desc

  ! single domain, multiple fields
  type(ghex_struct_domain)               :: domain_desc
  type(ghex_struct_communication_object) :: co
  type(ghex_struct_exchange_descriptor)  :: ed

  ! one field per domain, multiple domains
  type(ghex_struct_domain),               dimension(:) :: domain_descs(nfields)
  type(ghex_struct_communication_object), dimension(:) :: cos(nfields)
  type(ghex_struct_exchange_descriptor),  dimension(:) :: eds(nfields)
  type(ghex_struct_exchange_handle)      :: eh

  if (command_argument_count() /= 6) then
     print *, "Usage: <benchmark> [grid size] [niters] [halo size] [rank dims :3] "
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

  ! rank grid dimensions
  call get_command_argument(4, arg)
  read(arg,*) gdim(1)
  call get_command_argument(5, arg)
  read(arg,*) gdim(2)
  call get_command_argument(6, arg)
  read(arg,*) gdim(3)

  ! init mpi
  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
  call mpi_comm_rank(mpi_comm_world, world_rank, mpi_err)
  call mpi_comm_size(mpi_comm_world, size, mpi_err)
  ! call mpi_dims_create(size, 3, gdim, mpi_err)
  ! call mpi_cart_create(mpi_comm_world, 3, gdim, [1, 1, 1], .true., C_CART,ierr)
  C_CART = mpi_comm_world

  call mpi_comm_rank(C_CART, rank, mpi_err)

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world)

  ! halo information
  halo(:) = 0
  halo(1:2) = mb
  halo(3:4) = mb
  halo(5:6) = mb

  if (rank==0) then
     print *, "halos: ", halo
  end if
  
  ! use 27 ranks to test all halo connectivities
  if (size /= product(gdim)) then
    print *, "Usage: this test must be executed with ", product(gdim), " mpi ranks"
    call exit(1)
  end if

  ! define the global index domain
  gfirst = [1, 1, 1]
  glast = gdim * ldim

  ! local indices in the rank index space
  call rank2coord(rank, rank_coord)

  ! compute neighbor information
  call init_mpi_nbors(rank_coord)

  ! ! define the local domain
  first = (rank_coord-1) * ldim + 1
  last  = first + ldim - 1
  call ghex_domain_init(domain_desc, rank, first, last, gfirst, glast)

  ! make individual copies for sequenced comm
  i = 1
  do while (i <= nfields)
    call ghex_domain_init(domain_descs(i), rank, first, last, gfirst, glast)
    i = i+1
  end do

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
    allocate(data_ptr(i)%ptr(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
    data_ptr(i)%ptr(xs:xe, ys:ye, zs:ze) = rank
    i = i+1
  end do

  ! ---- COMPACT tests ----
  ! initialize the field datastructure
  i = 1
  do while (i <= nfields)
    call ghex_field_init(field_desc, data_ptr(i)%ptr, halo, periodic=[1,1,1])
    call ghex_domain_add_field(domain_desc, field_desc)
    call ghex_free(field_desc)
    i = i+1
  end do

  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

  ! create communication object
  call ghex_co_init(co)

  ! exchange halos
  eh = ghex_exchange(co, ed)
  call ghex_wait(eh)
  call cpu_time(tic)
  it = 0
  do while (it < niters)
    eh = ghex_exchange(co, ed)
    call ghex_wait(eh)
    it = it+1
  end do
  call cpu_time(toc)
  if (rank == 0) then 
     print *, rank, " exchange compact:      ", (toc-tic)
  end if

  call ghex_free(co)
  call ghex_free(ed)
  call ghex_free(domain_desc)

  ! ---- SEQUENCE tests ----
  ! initialize the field datastructure
  ! compute the halo information for all domains and fields
  i = 1
  do while (i <= nfields)
    call ghex_field_init(field_desc, data_ptr(i)%ptr, halo, periodic=[1,1,1])
    call ghex_domain_add_field(domain_descs(i), field_desc)
    call ghex_free(field_desc)
    eds(i) = ghex_exchange_desc_new(domain_descs(i))
    i = i+1
  end do
  
  ! create communication objects
  i = 1
  do while (i <= nfields)
    call ghex_co_init(cos(i))
    i = i+1
  end do

  ! exchange halos
  i = 1
  do while (i <= nfields)
    eh = ghex_exchange(cos(i), eds(i)); call ghex_wait(eh)
    i = i+1
  end do

  call cpu_time(tic)
  it = 0
  do while (it < niters)
    i = 1
    do while (i <= nfields)
      eh = ghex_exchange(cos(i), eds(i)); call ghex_wait(eh)
      i = i+1
    end do
    it = it+1
  end do
  call cpu_time(toc)
  if (rank == 0) then 
     print *, rank, " exchange sequenced (multiple COs):      ", (toc-tic); 
  end if


  ! ---- SEQUENCE tests, single CO ----
  ! exchange halos - SEQUENCE
  i = 1
  do while (i <= nfields)
    eh = ghex_exchange(co, eds(i)); call ghex_wait(eh)
    i = i+1
  end do

  call cpu_time(tic)
  it = 0
  do while (it < niters)
    i = 1
    do while (i <= nfields)
      eh = ghex_exchange(co, eds(i)); call ghex_wait(eh)
      i = i+1
    end do
    it = it+1
  end do
  call cpu_time(toc)
  if (rank == 0) then 
     print *, rank, " exchange sequenced (single CO):      ", (toc-tic); 
  end if

  ! ---- BIFROST-like comm ----

  call exchange_subarray_init

  i = 1
  do while (i <= nfields)
     call exchange_subarray(data_ptr(i)%ptr)
    i = i+1
  end do

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
  call cpu_time(toc)
  if (rank == 0) then 
     print *, rank, " subarray exchange (sendrecv):      ", (toc-tic)  
  end if

  i = 1
  do while (i <= nfields)
    call update_sendrecv(data_ptr(i)%ptr)
    i = i+1
  end do

  call cpu_time(tic)
  it = 0
  do while (it < niters)
    i = 1
    do while (i <= nfields)
      call update_sendrecv(data_ptr(i)%ptr)
      i = i+1
    end do
    it = it+1
  end do
  call cpu_time(toc)
  if (rank == 0) then 
     print *, rank, " bifrost exchange (sendrecv):      ", (toc-tic)  
  end if

  i = 1
  do while (i <= nfields)
    call update_sendrecv_2(data_ptr(i)%ptr)
    i = i+1
  end do

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
  call cpu_time(toc)
  if (rank == 0) then 
     print *, rank, " bifrost exchange 2 (sendrecv):      ", (toc-tic)  
  end if

  call mpi_barrier(mpi_comm_world, mpi_err)

  ! cleanup 
  call ghex_free(co)
  i = 1
  do while (i <= nfields)
    call ghex_free(domain_descs(i))
    call ghex_free(cos(i))
    call ghex_free(eds(i))
    deallocate(data_ptr(i)%ptr)
    i = i+1
  end do
  
  call ghex_finalize()
  call mpi_finalize(mpi_err)

contains
  
  ! -------------------------------------------------------------
  ! cartesian coordinates computations
  ! -------------------------------------------------------------
  subroutine rank2coord(rank, coord)
    integer :: rank, tmp, ierr
    integer :: coord(3)

    if (C_CART == mpi_comm_world) then
       tmp = rank;        coord(1) = modulo(tmp, gdim(1))
       tmp = tmp/gdim(1); coord(2) = modulo(tmp, gdim(2))
       tmp = tmp/gdim(2); coord(3) = tmp
    else
       call mpi_cart_coords(C_CART, rank, 3, coord, ierr)
    end if
    coord = coord + 1
  end subroutine rank2coord
  
  subroutine coord2rank(icoord, rank)
    integer :: icoord(3), coord(3), ierr
    integer :: rank

    coord = icoord - 1
    if (C_CART == mpi_comm_world) then
       rank = (coord(3)*gdim(2) + coord(2))*gdim(1) + coord(1)
    else
       call mpi_cart_rank(C_CART, coord, rank, ierr)
    end if
  end subroutine coord2rank
  
  function get_nbor(icoord, shift, idx)
    integer, intent(in) :: icoord(3)
    integer :: shift, idx
    integer :: get_nbor
    integer :: coord(3)
    
    coord = icoord
    coord(idx) = coord(idx)+shift
    if (C_CART == mpi_comm_world) then
       if (coord(idx) > gdim(idx)) then
          coord(idx) = 1
       end if
       if (coord(idx) == 0) then
          coord(idx) = gdim(idx)
       end if
    end if
    call coord2rank(coord, get_nbor)
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

END PROGRAM
