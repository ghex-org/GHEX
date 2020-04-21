PROGRAM test_halo_exchange
  use omp_lib
  use ghex_mod

  implicit none  

  include 'mpif.h'  

  real    :: tic, toc
  integer :: ierr, mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size, world_rank
  integer :: tmp, i, it
  integer :: gfirst(3), glast(3)       ! global index space
  integer :: first(3), last(3)
  integer :: gdim(3) = [2, 4, 2]       ! number of domains
  integer :: ldim(3) = [64, 64, 64]    ! dimensions of the local domains
  integer :: rank_coord(3)             ! local rank coordinates in a cartesian rank space
  integer :: halo(6)                   ! halo definition
  integer :: niters = 100

  ! -------------- variables used by the Bifrost-like implementation
  integer :: xsb, xeb, ysb, yeb, zsb, zeb
  integer :: xs , xe , ys , ye , zs , ze
  integer :: xr , xrb, yr , yrb, zr , zrb, mb
  integer :: C_CART, R_XUP, R_XDN, R_YUP, R_YDN, R_ZUP, R_ZDN
  logical :: exist_xup, exist_xdn, exist_yup, exist_ydn, exist_zup, exist_zdn
  integer(kind=4),dimension(MPI_STATUS_SIZE) :: status
  ! --------------   

  type hptr
     real(ghex_fp_kind), dimension(:,:,:), pointer :: ptr
  end type hptr
  
  ! exchange 8 data cubes
  type(hptr) :: data_ptr(8)

  ! GHEX stuff
  type(ghex_struct_field)                :: field_desc

  ! single domain, multiple fields
  type(ghex_struct_domain)               :: domain_desc
  type(ghex_struct_communication_object) :: co
  type(ghex_struct_exchange_descriptor)  :: ed

  ! one field per domain, multiple domains
  type(ghex_struct_domain),               dimension(:) :: domain_descs(8)
  type(ghex_struct_communication_object), dimension(:) :: cos(8)
  type(ghex_struct_exchange_descriptor),  dimension(:) :: eds(8)
  type(ghex_struct_exchange_handle)      :: eh

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

  ! halo width
  mb = 1
  
  ! halo information
  halo(:) = 0
  if (gdim(1) >1) then
    halo(1:2) = mb
  end if
  if (gdim(2) >1) then
    halo(3:4) = mb
  end if
  if (gdim(3) >1) then
    halo(5:6) = mb
  end if

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

  ! define the local domain
  first = (rank_coord-1) * ldim + 1
  last  = first + ldim - 1
  call ghex_domain_init(domain_desc, rank, first, last, gfirst, glast)

  ! make individual copies for sequenced comm
  i = 1
  do while (i <= 8)
    call ghex_domain_init(domain_descs(i), rank, first, last, gfirst, glast)
    i = i+1
  end do

  ! define local index ranges
  xsb = domain_desc%first(1) - halo(1)
  xeb = domain_desc%last(1)  + halo(2)
  ysb = domain_desc%first(2) - halo(3)
  yeb = domain_desc%last(2)  + halo(4)
  zsb = domain_desc%first(3) - halo(5)
  zeb = domain_desc%last(3)  + halo(6)

  xs  = domain_desc%first(1)
  xe  = domain_desc%last(1) 
  ys  = domain_desc%first(2)
  ye  = domain_desc%last(2) 
  zs  = domain_desc%first(3)
  ze  = domain_desc%last(3)

  xr  = xe  - xs
  yr  = ye  - ys
  zr  = ze  - zs
  xrb = xeb - xsb
  yrb = yeb - ysb
  zrb = zeb - zsb

  ! allocate and initialize data cubes
  i = 1
  do while (i <= 8)
    allocate(data_ptr(i)%ptr(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
    data_ptr(i)%ptr(xs:xe, ys:ye, zs:ze) = rank
    i = i+1
  end do


  ! ---- COMPACT tests ----
  ! initialize the field datastructure
  i = 1
  do while (i <= 8)
    call ghex_field_init(field_desc, data_ptr(i)%ptr, halo, periodic=[1,1,1])
    call ghex_domain_add_field(domain_desc, field_desc)
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

  call ghex_free(ed)
  call ghex_free(domain_desc)

  
  ! ---- SEQUENCE tests ----
  ! initialize the field datastructure
  ! compute the halo information for all domains and fields
  i = 1
  do while (i <= 8)
    call ghex_field_init(field_desc, data_ptr(i)%ptr, halo, periodic=[1,1,1])
    call ghex_domain_add_field(domain_descs(i), field_desc)
    eds(i) = ghex_exchange_desc_new(domain_descs(i))
    i = i+1
  end do
  
  ! create communication objects
  i = 1
  do while (i <= 8)
    call ghex_co_init(cos(i))
    i = i+1
  end do

  ! exchange halos
  i = 1
  do while (i <= 8)
    eh = ghex_exchange(cos(i), eds(i)); call ghex_wait(eh)
    i = i+1
  end do

  call cpu_time(tic)
  it = 0
  do while (it < niters)
    i = 1
    do while (i <= 8)
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
  do while (i <= 8)
    eh = ghex_exchange(co, eds(i)); call ghex_wait(eh)
    i = i+1
  end do

  call cpu_time(tic)
  it = 0
  do while (it < niters)
    i = 1
    do while (i <= 8)
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
  i = 1
  do while (i <= 8)
    call update_sendrecv(data_ptr(i)%ptr)
    i = i+1
  end do

  call cpu_time(tic)
  it = 0
  do while (it < niters)
    i = 1
    do while (i <= 8)
      call update_sendrecv(data_ptr(i)%ptr)
      i = i+1
    end do
    it = it+1
  end do
  call cpu_time(toc)
  if (rank == 0) then 
     print *, rank, " bifrost exchange (sendrecv):      ", (toc-tic)  
  end if

  call mpi_barrier(mpi_comm_world, mpi_err)

  ! cleanup 
  call ghex_free(co)
  i = 1
  do while (i <= 8)
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
       tmp = rank;        coord(3) = modulo(tmp, gdim(3))
       tmp = tmp/gdim(3); coord(2) = modulo(tmp, gdim(2))
       tmp = tmp/gdim(2); coord(1) = tmp
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
       rank = (coord(1)*gdim(2) + coord(2))*gdim(3) + coord(3)
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
    integer :: tmp(3)

    exist_xup = .false.
    exist_xdn = .false.
    exist_yup = .false.
    exist_ydn = .false.
    exist_zup = .false.
    exist_zdn = .false.

    ! x dimension is periodic
    if (gdim(1) > 1) then
      R_XUP = get_nbor(rank_coord, +1, 1); exist_xup = .true.
      R_XDN = get_nbor(rank_coord, -1, 1); exist_xdn = .true.
    else
      R_XUP = MPI_PROC_NULL
      R_XDN = MPI_PROC_NULL
    end if

    ! y dimension is periodic
    if (gdim(2) > 1) then
      R_YUP = get_nbor(rank_coord, +1, 2); exist_yup = .true.
      R_YDN = get_nbor(rank_coord, -1, 2); exist_ydn = .true.
    else
      R_YUP = MPI_PROC_NULL
      R_YDN = MPI_PROC_NULL
    end if

    ! z dimension is NOT periodic
    ! if (gdim(3) > 1) then
    !   if (rank_coord(3) /= gdim(3)) then
    !     R_ZUP = get_nbor(rank_coord, +1, 3); exist_zup = .true.
    !   else
    !     R_ZUP = MPI_PROC_NULL
    !   end if
    !   if (rank_coord(3) /= 1) then
    !     R_ZDN = get_nbor(rank_coord, -1, 3); exist_zdn = .true.
    !   else
    !     R_ZDN = MPI_PROC_NULL
    !   end if
    ! end if
    if (gdim(3) > 1) then
       R_ZUP = get_nbor(rank_coord, +1, 3); exist_zup = .true.
       R_ZDN = get_nbor(rank_coord, -1, 3); exist_zdn = .true.
    else
       R_ZUP = MPI_PROC_NULL
       R_ZDN = MPI_PROC_NULL
    end if

    ! print *, rank, R_XUP, R_XDN, R_YUP, R_YDN, R_ZUP, R_ZDN

  end subroutine init_mpi_nbors
  
 
  ! -------------------------------------------------------------
  ! Bifrost-like communication with 3 synchroneous steps
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

    if (exist_xup.or.exist_xdn) then

      call MPI_SENDRECV(f(xe-(mb-1) :xe         , &
        ys        :ye         , &
        zs        :ze        ), &
        mb*yr*zr,MPI_REAL,R_XUP,1  , &
        f(xsb     :xsb+(mb-1) , &
        ys        :ye         , &
        zs        :ze        ), &
        mb*yr*zr,MPI_REAL,R_XDN,1  , &
        C_CART,status,ierr)

      call MPI_SENDRECV(f(xs        :xs+(mb-1)  , &
        ys        :ye         , &
        zs        :ze        ), &
        mb*yr*zr,MPI_REAL,R_XDN,2  , &
        f(xeb-(mb-1):xeb        , &
        ys        :ye         , &
        zs        :ze        ), &
        mb*yr*zr,MPI_REAL,R_XUP,2  , &
        C_CART,status,ierr)

    end if
  end subroutine comm_x

  subroutine comm_y(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    if (exist_yup.or.exist_ydn) then

      call MPI_SENDRECV(f(xsb       :xeb              , &
        ye-(mb-1):ye              , &
        zs       :ze       )      , &
        xrb*mb*zr,MPI_REAL,R_YUP,3, &
        f(xsb       :xeb              , &
        ysb         :ysb+(mb-1)         , &
        zs       :ze       )      , &
        xrb*mb*zr,MPI_REAL,R_YDN,3, &
        C_CART,status,ierr)


      call MPI_SENDRECV(f(xsb       :xeb              , &
        ys       :ys+(mb-1)       , &
        zs       :ze       )      , &
        xrb*mb*zr,MPI_REAL,R_YDN,4          , &
        f(xsb       :xeb              , &
        yeb-(mb-1)  :yeb                , &
        zs       :ze       )      , &
        xrb*mb*zr,MPI_REAL,R_YUP,4           , &
        C_CART,status,ierr)

    end if
  end subroutine comm_y

  subroutine comm_z(f)
    implicit none

    real(kind=4),dimension(xsb:xeb,ysb:yeb,zsb:zeb) :: f

    if (exist_zdn.or.exist_zup) then

      call MPI_SENDRECV( &
        f(xsb        :xeb              , &
        ysb        :yeb              , &
        ze-(mb-1) :ze          )   , &
        xrb*yrb*mb,MPI_REAL,R_ZUP,5, &
        f(xsb        :xeb              , &
        ysb        :yeb              , &
        zsb       :zsb+(mb-1)  )   , &
        xrb*yrb*mb,MPI_REAL,R_ZDN,5, &
        C_CART,status,ierr)


      call MPI_SENDRECV( &
        f(xsb        :xeb              , &
        ysb        :yeb              , &
        zs        :zs+(mb-1))      , &
        xrb*yrb*mb,MPI_REAL,R_ZDN,6, &
        f(xsb        :xeb              , &
        ysb        :yeb              , &
        zeb-(mb-1):zeb         )   , &
        xrb*yrb*mb,MPI_REAL,R_ZUP,6, &
        C_CART,status,ierr)

    end if
  end subroutine comm_z

END PROGRAM
