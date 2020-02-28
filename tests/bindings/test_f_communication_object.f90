PROGRAM test_halo_exchange
  use omp_lib
  use ghex_mod
  use ghex_structured_mod
  use ghex_exchange_mod

  implicit none  

  include 'mpif.h'  

  real    :: tic, toc
  integer :: ierr, mpi_err, mpi_threading
  integer :: nthreads = 1, rank, size
  integer :: tmp, i
  integer :: gfirst(3), glast(3)      ! global index space
  integer :: gdim(3) = [1, 2, 2]      ! number of domains
  integer :: ldim(3) = [128, 64, 64]  ! dimensions of the local domains
  integer :: rank_coord(3)            ! local rank coordinates in a cartesian rank space
  integer :: halo(6)                  ! halo definition
  integer :: niters = 1000

  ! -------------- variables used by the Bifrost-like implementation
  integer :: xsb, xeb, ysb, yeb, zsb, zeb
  integer :: xs , xe , ys , ye , zs , ze
  integer :: xr , xrb, yr , yrb, zr , zrb, mb
  integer :: C_CART, R_XUP, R_XDN, R_YUP, R_YDN, R_ZUP, R_ZDN
  logical :: exist_xup, exist_xdn, exist_yup, exist_ydn, exist_zup, exist_zdn
  integer(kind=4),dimension(MPI_STATUS_SIZE) :: status
  ! --------------   

  ! exchange 8 data cubes
  real(ghex_fp_kind), dimension(:,:,:), pointer :: data1, data2, data3, data4
  real(ghex_fp_kind), dimension(:,:,:), pointer :: data5, data6, data7, data8

  ! GHEX stuff
  type(ghex_domain_descriptor), target, dimension(:) :: domain_desc(1), d1(1), d2(1), d3(1), d4(1), d5(1), d6(1), d7(1), d8(1)
  type(ghex_field_descriptor)     :: field_desc
  type(ghex_communication_object) :: co, co1, co2, co3, co4, co5, co6, co7, co8
  type(ghex_exchange_descriptor)  :: ed, ed1, ed2, ed3, ed4, ed5, ed6, ed7, ed8
  type(ghex_exchange_handle)      :: ex_handle

  ! init mpi
  call mpi_init_thread (MPI_THREAD_SINGLE, mpi_threading, mpi_err)
  call mpi_comm_rank(mpi_comm_world, rank, mpi_err)
  call mpi_comm_size(mpi_comm_world, size, mpi_err)
  C_CART = mpi_comm_world

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world)

  ! halo width
  mb = 5
  
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
  domain_desc(1)%id = rank
  domain_desc(1)%device_id = DeviceCPU
  domain_desc(1)%first = (rank_coord-1) * ldim + 1
  domain_desc(1)%last  = domain_desc(1)%first + ldim - 1
  domain_desc(1)%gfirst = gfirst
  domain_desc(1)%glast  = glast

  ! make individual copies for sequenced comm
  call copy_domain(d1, domain_desc)
  call copy_domain(d2, domain_desc)
  call copy_domain(d3, domain_desc)
  call copy_domain(d4, domain_desc)
  call copy_domain(d5, domain_desc)
  call copy_domain(d6, domain_desc)
  call copy_domain(d7, domain_desc)
  call copy_domain(d8, domain_desc)

  ! define local index ranges
  xsb = domain_desc(1)%first(1) - halo(1)
  xeb = domain_desc(1)%last(1) + halo(2)
  ysb = domain_desc(1)%first(2) - halo(3)
  yeb = domain_desc(1)%last(2) + halo(4)
  zsb = domain_desc(1)%first(3) - halo(5)
  zeb = domain_desc(1)%last(3) + halo(6)

  xs  = domain_desc(1)%first(1)
  xe  = domain_desc(1)%last(1) 
  ys  = domain_desc(1)%first(2)
  ye  = domain_desc(1)%last(2) 
  zs  = domain_desc(1)%first(3)
  ze  = domain_desc(1)%last(3)

  xr  = xe  - xs
  yr  = ye  - ys
  zr  = ze  - zs
  xrb = xeb - xsb
  yrb = yeb - ysb
  zrb = zeb - zsb

  ! allocate and initialize field data
  allocate(data1(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data2(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data3(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data4(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data5(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data6(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data7(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)
  allocate(data8(xsb:xeb, ysb:yeb, zsb:zeb), source=-1.0)

  data1(xs:xe, ys:ye, zs:ze) = rank

  call cpu_time(tic)
  i = 0
  do while (i < niters)
    call update_sendrecv(data1)
    call update_sendrecv(data2)
    call update_sendrecv(data3)
    call update_sendrecv(data4)
    call update_sendrecv(data5)
    call update_sendrecv(data6)
    call update_sendrecv(data7)
    call update_sendrecv(data8)
    i = i + 1
  end do
  call cpu_time(toc)
  print *, rank, " bifrost exchange (sendrecv):      ", (toc-tic)  
  
  ! initialize the field datastructure - COMPACT
  call ghex_field_init(field_desc, data1, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data2, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data3, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data4, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data5, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data6, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data7, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  call ghex_field_init(field_desc, data8, halo, periodic=[1,1,0])
  call ghex_domain_add_field(domain_desc(1), field_desc)

  ! compute the halo information for all domains and fields
  ed = ghex_exchange_desc_new(domain_desc)

  ! initialize the field datastructure - SEQUENCE  
  call ghex_field_init(field_desc, data1, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d1(1), field_desc)
  ed1 = ghex_exchange_desc_new(d1)

  call ghex_field_init(field_desc, data2, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d2(1), field_desc)
  ed2 = ghex_exchange_desc_new(d2)

  call ghex_field_init(field_desc, data3, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d3(1), field_desc)
  ed3 = ghex_exchange_desc_new(d3)

  call ghex_field_init(field_desc, data4, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d4(1), field_desc)
  ed4 = ghex_exchange_desc_new(d4)

  call ghex_field_init(field_desc, data5, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d5(1), field_desc)
  ed5 = ghex_exchange_desc_new(d5)

  call ghex_field_init(field_desc, data6, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d6(1), field_desc)
  ed6 = ghex_exchange_desc_new(d6)

  call ghex_field_init(field_desc, data7, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d7(1), field_desc)
  ed7 = ghex_exchange_desc_new(d7)

  call ghex_field_init(field_desc, data8, halo, periodic=[1,1,0])
  call ghex_domain_add_field(d8(1), field_desc)
  ed8 = ghex_exchange_desc_new(d8)

  ! create communication object
  co = ghex_struct_co_new()

  ! exchange halos - COMPACT
  call cpu_time(tic)
  i = 0
  do while (i < niters)
    ex_handle = ghex_exchange(co, ed)
    call ghex_wait(ex_handle)
    call ghex_delete(ex_handle)
    i = i+1
  end do
  call cpu_time(toc)
  print *, rank, " exchange compact:      ", (toc-tic)

  co1 = ghex_struct_co_new()
  co2 = ghex_struct_co_new()
  co3 = ghex_struct_co_new()
  co4 = ghex_struct_co_new()
  co5 = ghex_struct_co_new()
  co6 = ghex_struct_co_new()
  co7 = ghex_struct_co_new()
  co8 = ghex_struct_co_new()

  ! exchange halos - SEQUENCE
  call cpu_time(tic)
  i = 0
  do while (i < niters)
    ex_handle = ghex_exchange(co, ed1); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed2); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed3); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed4); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed5); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed6); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed7); call ghex_wait(ex_handle)
    ex_handle = ghex_exchange(co, ed8); call ghex_wait(ex_handle)
    i = i+1
  end do
  call cpu_time(toc)
  print *, rank, " exchange compact:      ", (toc-tic)

  ! cleanup
  call ghex_delete(ed)
  call ghex_delete(co)
  call ghex_delete(domain_desc(1))
  call ghex_finalize()
  call mpi_finalize(mpi_err)

contains

  subroutine copy_domain(dst, src)
    type(ghex_domain_descriptor), intent(inout) :: dst(1)
    type(ghex_domain_descriptor), intent(in) :: src(1)
    dst(1)%id = rank
    dst(1)%device_id = DeviceCPU
    dst(1)%first(:)  = src(1)%first(:)
    dst(1)%last(:)   = src(1)%last(:)
    dst(1)%gfirst(:) = src(1)%gfirst(:)
    dst(1)%glast(:)  = src(1)%glast(:)
  end subroutine copy_domain

  
  ! -------------------------------------------------------------
  ! cartesian coordinates computations
  ! -------------------------------------------------------------
  subroutine rank2coord(rank, coord)
    integer :: rank, tmp
    integer :: coord(3)

    tmp = rank;        coord(1) = modulo(tmp, gdim(1))
    tmp = tmp/gdim(1); coord(2) = modulo(tmp, gdim(2))
    tmp = tmp/gdim(2); coord(3) = tmp
    coord = coord + 1
  end subroutine rank2coord
  
  subroutine coord2rank(icoord, rank)
    integer :: icoord(3), coord(3)
    integer :: rank

    coord = icoord - 1
    rank = (coord(3)*gdim(2) + coord(2))*gdim(1) + coord(1)
  end subroutine coord2rank
  
  function get_nbor(icoord, shift, idx)
    integer, intent(in) :: icoord(3)
    integer :: shift, idx
    integer :: get_nbor
    integer :: coord(3)
    
    coord = icoord
    coord(idx) = coord(idx)+shift
    if (coord(idx) > gdim(idx)) then
      coord(idx) = 1
    end if
    if (coord(idx) == 0) then
      coord(idx) = gdim(idx)
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
    if (gdim(3) > 1) then
      if (rank_coord(3) /= gdim(3)) then
        R_ZUP = get_nbor(rank_coord, +1, 3); exist_zup = .true.
      else
        R_ZUP = MPI_PROC_NULL
      end if
      if (rank_coord(3) /= 1) then
        R_ZDN = get_nbor(rank_coord, -1, 3); exist_zdn = .true.
      else
        R_ZDN = MPI_PROC_NULL
      end if
    end if
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
