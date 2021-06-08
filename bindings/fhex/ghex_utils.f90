MODULE ghex_utils
  use mpi
  use ghex_defs
  implicit none

  interface
     integer(c_int) function ghex_get_current_cpu() bind(c)
       use iso_c_binding
     end function ghex_get_current_cpu

     integer(c_int) function ghex_get_ncpus() bind(c)
       use iso_c_binding
     end function ghex_get_ncpus
  end interface

contains


  ! obtain cartesian coordinates of the calling rank for all topological levels
  recursive subroutine ghex_cart_topology(comm, domain, topo, level_rank, ilevel)
    implicit none
    integer(kind=4), intent(in)  :: comm
    integer(kind=4), intent(in)  :: domain(:)
    integer(kind=4), intent(in)  :: topo(:,:)
    integer(kind=4), intent(out) :: level_rank(:)
    integer(kind=4), intent(in), optional  :: ilevel
    
    integer(kind=4) :: level
    integer(kind=4), dimension(:), allocatable :: sbuff, rbuff, group
    integer(kind=4) :: buffer(1)
    integer(kind=4) :: ierr, comm_rank, comm_size, nodeid, noderank, color, ii
    integer(kind=4) :: split_comm, split_type, split_rank, split_size
    integer(kind=4) :: master_comm, master_size

    level = size(level_rank)
    if (present(ilevel)) level = ilevel

    ! parent communicator
    call MPI_Comm_rank(comm, comm_rank, ierr)
    call MPI_Comm_size(comm, comm_size, ierr)

    if (level == 1) then
       ! we've reached the bottom of the topology
       ! verify topology validity on this level
       if (comm_size /= product(topo(:,level))) then
          write (*,"(a,a,i4,a,i4)") "ERROR: wrong topology on botton level", \
          ": expected", product(topo(:,level)), " domains (config), but found", comm_size, " (hardware)"
          call exit(1)
       end if
       level_rank(level) = comm_rank
       return
    end if

    ! split the current communicator to a lower level
    split_type = domain(level-1)

    ! create communicator for this topology level
    call MPI_Comm_split_type(comm, split_type, 0, MPI_INFO_NULL, split_comm, ierr)
    call MPI_Comm_rank(split_comm, split_rank, ierr)
    call MPI_Comm_size(split_comm, split_size, ierr)

    ! no split on this topology level
    if (split_size == comm_size) then
       if (1 /= product(topo(:,level))) then
          write (*,"(a,i4,a,i4,a,i4)") "ERROR (2): wrong topology on level", level, \
          ": expected", product(topo(:,level)), " domains (config), but found", comm_size, " (hardware)"
          call exit(1)
       end if
       level_rank(level) = 0
       call ghex_cart_topology(split_comm, domain, topo, level_rank, level-1)
       return
    end if
    
    
    ! make a master-rank communicator: masters from each split comm join
    color = 0
    if (split_rank /= 0) then
       ! non-masters
       call MPI_Comm_split(comm, color, 0, master_comm, ierr)
    else
       ! masters
       ! temporary nodeid identifier: rank of the split master
       nodeid = comm_rank+1
       color = 1
       call MPI_Comm_split(comm, color, 0, master_comm, ierr)
       call MPI_Comm_size(master_comm, master_size, ierr)

       ! verify topology validity on this level
       if (master_size /= product(topo(:,level))) then
          write (*,"(a,i4,a,i4,a,i4,a)") "ERROR (3): wrong topology on level", level, \
          ": expected", product(topo(:,level)), " domains (config), but found", master_size, " (hardware)"
          call exit(1)
       end if

       ! comm buffers to establish unique node id's for each master
       allocate(sbuff(1:comm_size), rbuff(1:comm_size), Source=0)

       ! find node rank based on unique node id from above
       sbuff(:) = nodeid
       call MPI_Alltoall(sbuff, 1, MPI_INT, rbuff, 1, MPI_INT, master_comm, ierr)
       
       ! mark each unique node id with a 1
       sbuff(:) = 0
       sbuff(pack(rbuff, rbuff/=0)) = 1

       ! cumsum: finds node rank for each unique node id
       ii = 1
       do while(ii<comm_size)
          sbuff(ii+1) = sbuff(ii+1) + sbuff(ii)
          ii = ii+1
       end do
       noderank = sbuff(nodeid) - 1

       ! cleanup
       deallocate(sbuff, rbuff)
    end  if
    
    call MPI_Comm_Disconnect(master_comm, ierr)

    ! distribute the node id to all split ranks
    buffer(1) = noderank
    call MPI_Bcast(buffer, 1, MPI_INT, 0, split_comm, ierr)

    ! save our level rank
    level_rank(level) = buffer(1)
    
    ! sub-divide lower levels
    call ghex_cart_topology(split_comm, domain, topo, level_rank, level-1)

    ! cleanup
    call MPI_Comm_Disconnect(split_comm, ierr)
  end subroutine ghex_cart_topology


  ! obtain compute node ID of the calling rank
  subroutine ghex_get_noderank(comm, noderank, isplit_type)
    implicit none
    integer(kind=4), intent(in)  :: comm
    integer(kind=4), intent(out) :: noderank
    integer(kind=4), intent(in), optional  :: isplit_type
    integer(kind=4), dimension(:), allocatable :: sbuff, rbuff
    integer(kind=4) :: nodeid, rank, size, ierr, ii, split_type
    integer(kind=4) :: shmcomm
    character(len=MPI_MAX_LIBRARY_VERSION_STRING) :: version
    integer :: resultlen

    split_type = MPI_COMM_TYPE_SHARED
    if (present(isplit_type)) split_type = isplit_type

    ! old rank
    call MPI_Comm_rank(comm, rank, ierr)
    call MPI_Comm_size(comm, size, ierr)

    ! communication buffers
    allocate(sbuff(1:size), rbuff(1:size), Source=0)

    ! check for OpenMPI to use fine-grain hwloc domains
    call MPI_Get_library_version(version, resultlen, ierr)
    ii = index(version, 'Open MPI')
    if (ii /= 0) then

       ! create local communicator
       call MPI_Comm_split_type(comm, split_type, 0, MPI_INFO_NULL, shmcomm, ierr)
    else
       
       ! create local communicator
       call MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, shmcomm, ierr)
    end if

    ! figure out unique compute node id = max rank id on each compute node
    sbuff(1) = rank+1
    call MPI_Allreduce(sbuff, rbuff, 1, MPI_INT, MPI_MAX, shmcomm, ierr)
    nodeid = rbuff(1)

    ! find node rank based on unique node id from above
    sbuff(:) = nodeid
    call MPI_Alltoall(sbuff, 1, MPI_INT, rbuff, 1, MPI_INT, comm, ierr)

    ! mark each unique node id with a 1
    sbuff(:) = 0
    sbuff(rbuff) = 1

    ! cumsum: finds node rank for each unique node id
    ii = 1
    do while(ii<size)
       sbuff(ii+1) = sbuff(ii+1) + sbuff(ii)
       ii = ii+1
    end do
    noderank = sbuff(nodeid) - 1

    ! cleanup
    deallocate(sbuff, rbuff)
    call MPI_Comm_Disconnect(shmcomm, ierr)

  end subroutine ghex_get_noderank


  ! obtain cartesian coordinates of rank.
  ! comm can be MPI_COMM_NULL
  ! if comm is an MPI cartesian communicator, we use MPI
  subroutine ghex_cart_rank2coord(comm, dims, rank, coord, iorder)
    implicit none
    integer(kind=4), intent(in)  :: comm, dims(3), rank
    integer(kind=4), intent(out) :: coord(3)
    integer(kind=4), optional :: iorder
    integer(kind=4) :: topo, ierr, tmp, order

    if (present(iorder)) then
       order = iorder
    else
       order = CartOrderDefault
    endif    

    ! check if this is a cartesian communicator
    if (comm == MPI_COMM_NULL) then
       topo = -1;
    else
       call MPI_Topo_test(comm, topo, ierr)
    end if
    if (topo/=MPI_CART) then

       select case (order)
          case (CartOrderXYZ)
             tmp = rank;        coord(1) = modulo(tmp, dims(1))
             tmp = tmp/dims(1); coord(2) = modulo(tmp, dims(2))
             tmp = tmp/dims(2); coord(3) = tmp
       
          case (CartOrderXZY)
             tmp = rank;        coord(1) = modulo(tmp, dims(1))
             tmp = tmp/dims(1); coord(3) = modulo(tmp, dims(3))
             tmp = tmp/dims(3); coord(2) = tmp

          case (CartOrderZYX)
             tmp = rank;        coord(3) = modulo(tmp, dims(3))
             tmp = tmp/dims(3); coord(2) = modulo(tmp, dims(2))
             tmp = tmp/dims(2); coord(1) = tmp

          case (CartOrderYZX)
             tmp = rank;        coord(2) = modulo(tmp, dims(2))
             tmp = tmp/dims(2); coord(3) = modulo(tmp, dims(3))
             tmp = tmp/dims(3); coord(1) = tmp

          case (CartOrderZXY)
             tmp = rank;        coord(3) = modulo(tmp, dims(3))
             tmp = tmp/dims(3); coord(1) = modulo(tmp, dims(1))
             tmp = tmp/dims(1); coord(2) = tmp

          case (CartOrderYXZ)
             tmp = rank;        coord(2) = modulo(tmp, dims(2))
             tmp = tmp/dims(2); coord(1) = modulo(tmp, dims(1))
             tmp = tmp/dims(1); coord(3) = tmp

          case default
             print *, "unknown value of argument 'order': ", order
             call exit
          end select
    else
       call mpi_cart_coords(comm, rank, 3, coord, ierr)
    end if

  end subroutine ghex_cart_rank2coord


  ! obtain rank from cartesian coordinates of a domain.
  ! handles periodicity, i.e., coord can have numbers <0 and >=dims
  ! comm can be MPI_COMM_NULL
  ! if comm is an MPI cartesian communicator, we use MPI
  subroutine ghex_cart_coord2rank(comm, dims, periodic, coord, rank, iorder)
    implicit none
    integer(kind=4), intent(in) :: comm, dims(3), coord(3)
    logical, intent(in) :: periodic(3)
    integer(kind=4), intent(out) :: rank
    integer(kind=4) :: tcoord(3), dim, topo, ierr, order
    integer(kind=4), optional :: iorder

    if (present(iorder)) then
       order = iorder
    else
       order = CartOrderDefault
    endif    

    ! apply periodicity
    dim = 1
    tcoord = coord
    do while (dim.le.size(dims))
       if (tcoord(dim)<0) then
          if (periodic(dim)) then
             tcoord(dim) = dims(dim)-1
          else
             rank = MPI_PROC_NULL
             return
          end if
       end if
       if (tcoord(dim)>=dims(dim)) then
          if (periodic(dim)) then
             tcoord(dim) = 0
          else
             rank = MPI_PROC_NULL
             return
          end if
       end if
       dim = dim + 1
    end do

    ! check if this is a cartesian communicator
    if (comm == MPI_COMM_NULL) then
       topo = -1;
    else
       call MPI_Topo_test(comm, topo, ierr)
    end if
    if (topo/=MPI_CART) then

       select case (order)
          case (CartOrderXYZ)
             rank = (tcoord(3)*dims(2) + tcoord(2))*dims(1) + tcoord(1)
       
          case (CartOrderXZY)
             rank = (tcoord(2)*dims(3) + tcoord(3))*dims(1) + tcoord(1)

          case (CartOrderZYX)
             rank = (tcoord(1)*dims(2) + tcoord(2))*dims(3) + tcoord(3)

          case (CartOrderYZX)
             rank = (tcoord(1)*dims(3) + tcoord(3))*dims(2) + tcoord(2)

          case (CartOrderZXY)
             rank = (tcoord(2)*dims(1) + tcoord(1))*dims(3) + tcoord(3)

          case (CartOrderYXZ)
             rank = (tcoord(3)*dims(1) + tcoord(1))*dims(2) + tcoord(2)

          case default
             print *, "unknown value of argument 'order': ", order
             call exit
          end select
    else
       call mpi_cart_rank(comm, tcoord, rank, ierr)
    end if

  end subroutine ghex_cart_coord2rank


  ! create a cartesian sub-communicator (as mpi_cart_sub)
  ! if comm is an MPI cartesian communicator, we use MPI
  subroutine ghex_cart_create_subcomm(comm, dims, rank, belongs, newcomm, iorder)
    implicit none
    integer(kind=4), intent(in)      :: comm, dims(3), rank
    logical,dimension(3), intent(in) :: belongs
    integer(kind=4), intent(out)     :: newcomm
    integer :: ierr, coord(3), color, topo, order
    integer(kind=4), optional :: iorder

    if (present(iorder)) then
       order = iorder
    else
       order = CartOrderDefault
    endif    

    ! check if this is a cartesian communicator
    if (comm == MPI_COMM_NULL) then
       topo = -1;
    else
       call MPI_Topo_test(comm, topo, ierr)
    end if
    if (topo/=MPI_CART) then

       ! Find ranks belonging to the new communicator based on each rank's cartesian coordinates.
       ! color is computed by zeroing out 'belongs' in the cartesian coordinate of each rank
       ! and then computing the 'collapsed' rank by coord2rank
       call ghex_cart_rank2coord(comm, dims, rank, coord, order)
       where (belongs)
          coord = 0
       end where
       call ghex_cart_coord2rank(comm, dims, (/.false., .false., .false./), coord, color, order)

       ! create the new communicator
       call MPI_Comm_split(comm, color, 0, newcomm, ierr)
    else
       call MPI_Cart_sub(comm, belongs, newcomm, ierr)
    end if

  end subroutine ghex_cart_create_subcomm

  ! Renumber the MPI ranks according to user-specified memory domain topology.
  !  - domain: specifies OPENMPI locality domains
  !
  ! A new communicator is created with this rank numbering.
  ! It can then be used as usual with rank2coord, coord2rank,
  ! and create_subcomm.
  subroutine ghex_cart_remap_ranks(comm, domain, topo, level_rank, newcomm, iorder)
    implicit none
    integer(kind=4), intent(in)  :: comm
    integer(kind=4), intent(in)  :: domain(:)
    integer(kind=4), intent(in)  :: topo(:,:)
    integer(kind=4), intent(in) :: level_rank(:)
    integer(kind=4), intent(out) :: newcomm
    integer(kind=4), optional :: iorder

    integer(kind=4) ::          topo_coord(3), gdim(3)
    integer(kind=4) ::          newrank, newcoord(3)
    integer(kind=4) :: shmcomm, shmrank, shmcoord(3), shmdims(3), shmsize, rank, size
    integer(kind=4) ::         noderank, nodecoord(3)
    integer(kind=4) :: ierr, order, ii, comm_rank
    integer(kind=4), allocatable, dimension(:,:) :: cartXYZ
    integer(kind=4), allocatable, dimension(:,:) :: dims

    call MPI_Comm_rank(comm, comm_rank, ierr)

    if (present(iorder)) then
       order = iorder
    else
       order = CartOrderDefault
    endif    

    allocate(cartXYZ(1:3,1:size(domain)))
    allocate(dims(1:3,1:size(domain)))

    ! compute rank cartesian coordinates for each topological level
    ii = 1
    do while(ii<=size(domain))
       call ghex_cart_rank2coord(MPI_COMM_NULL, topo(:,ii), level_rank(ii), cartXYZ(:,ii), order)
       ii = ii+1
    end do

    ! assemble to global cartesian coordinates
    dims = cshift(topo, -1, dim=2)
    dims(:,1) = 1
    ii = 2
    do while(ii<=size(domain))
       dims(:,ii) = dims(:,ii)*dims(:,ii-1)
       ii = ii+1
    end do
    topo_coord = sum(dims*cartXYZ, 2)

    ! compute global grid dimensions
    gdim = product(topo, dim=2)

    ! compute rank id in global ranks space
    call ghex_cart_coord2rank(comm, gdim, (/.false., .false., .false./), topo_coord, newrank, order)
    
    ! create the new communicator with remapped ranks
    call MPI_Comm_split(comm, 0, newrank, newcomm, ierr)
  end subroutine ghex_cart_remap_ranks
  

  subroutine ghex_cart_print_rank2node(comm, dims, iorder)
    implicit none
    integer(kind=4), intent(in) :: comm, dims(3)
    integer(kind=4), dimension(:,:), allocatable :: buff
    integer(kind=4) :: sbuff(4), ierr
    integer(kind=4) :: rank, size, noderank, order, orank, ncpus
    character(len=20) :: fmt, fmti
    integer(kind=4), optional :: iorder

    if (present(iorder)) then
       order = iorder
    else
       order = CartOrderDefault
    endif    

    call ghex_get_noderank(comm, noderank)

    ! obtain all values at master
    call MPI_Comm_rank(mpi_comm_world, orank, ierr)
    call MPI_Comm_rank(comm, rank, ierr)
    call MPI_Comm_size(comm, size, ierr)
    allocate(buff(1:4,0:size-1), Source=0)
    sbuff(1) = rank
    sbuff(2) = orank
    sbuff(3) = ghex_get_current_cpu()
    sbuff(4) = noderank

    ! assuming HT is enabled, use the lower core ID
    ncpus = ghex_get_ncpus()/2
    if (sbuff(3) >= ncpus) then
       sbuff(3) = sbuff(3) - ncpus
    end if
    call MPI_Gather(sbuff, 4, MPI_INT, buff, 4, MPI_INT, 0, comm, ierr)

    if (rank==0) then
       write (*,*) ' '
       write (*,*) ' Rank to node mapping '
       write (*,*) ' '
       call print_cube(comm, dims, 4, buff, order)

       write (*,*) ' '
       write (*,*) ' Rank layout '
       write (*,*) ' '
       call print_cube(comm, dims, 1, buff, order)

       write (*,*) ' '
       write (*,*) ' MPI_COMM_WORLD layout '
       write (*,*) ' '
       call print_cube(comm, dims, 2, buff, order)

       write (*,*) ' '
       write (*,*) ' CPU '
       write (*,*) ' '
       call print_cube(comm, dims, 3, buff, order)

       write (*,*) ' '
       write (*,*) 'Z |    / Y'
       write (*,*) '  |   /'
       write (*,*) '  |  /'
       write (*,*) '  | /'
       write (*,*) '  |_______ X'
       write (*,*) ' '
    end if

    deallocate(buff)

  end subroutine ghex_cart_print_rank2node


  subroutine ghex_cart_print_rank_topology(comm, domain, topo, iorder)
    implicit none
    integer(kind=4), intent(in) :: comm
    integer(kind=4), intent(in) :: domain(:), topo(:,:)
    integer(kind=4), optional :: iorder

    integer(kind=4), dimension(:,:), allocatable :: buff
    integer(kind=4) :: ierr, li, dims(3), nlevels
    integer(kind=4) :: comm_rank, comm_size, order, orank, ncpus
    integer(kind=4), dimension(:), allocatable :: sbuff, level_rank
    character(len=32) :: name

    ! compute global grid dimensions
    dims = product(topo, dim=2)

    if (present(iorder)) then
       order = iorder
    else
       order = CartOrderDefault
    endif    

    nlevels = size(domain)-1
    allocate(sbuff(1:nlevels+3), level_rank(1:nlevels), source=0)

    li = 1
    do while (li<=nlevels)
       call ghex_get_noderank(comm, level_rank(li), domain(li))
       li = li+1
    end do

    ! obtain all values at master
    call MPI_Comm_rank(mpi_comm_world, orank, ierr)
    call MPI_Comm_rank(comm, comm_rank, ierr)
    call MPI_Comm_size(comm, comm_size, ierr)
    allocate(buff(1:size(sbuff),0:comm_size-1), source=0)
    sbuff(1) = comm_rank
    sbuff(2) = orank
    sbuff(3) = ghex_get_current_cpu()
    sbuff(4:) = level_rank

    ! assuming HT is enabled, use the lower core ID
    ncpus = ghex_get_ncpus()/2
    if (sbuff(3) >= ncpus) then
       sbuff(3) = sbuff(3) - ncpus
    end if
    call MPI_Gather(sbuff, size(sbuff), MPI_INT, buff, size(sbuff), MPI_INT, 0, comm, ierr)

    if (comm_rank==0) then
       write (*,*) ' '
       write (*,*) ' Rank to node mapping '
       write (*,*) ' '

       li = 1
       do while(li<=nlevels)
          call split_type_to_name(domain(li), name)
          write (*,*) ' '
          write (*,'(a,i4,a,a)') ' Level ', li, ' ', name
          write (*,*) ' '
          call print_cube(comm, dims, 3+li, buff, order)
          li = li+1
       end do

       write (*,*) ' '
       write (*,*) ' Rank layout '
       write (*,*) ' '
       call print_cube(comm, dims, 1, buff, order)

       write (*,*) ' '
       write (*,*) ' MPI_COMM_WORLD layout '
       write (*,*) ' '
       call print_cube(comm, dims, 2, buff, order)

       write (*,*) ' '
       write (*,*) ' CPU '
       write (*,*) ' '
       call print_cube(comm, dims, 3, buff, order)

       write (*,*) ' '
       write (*,*) 'Z |    / Y'
       write (*,*) '  |   /'
       write (*,*) '  |  /'
       write (*,*) '  | /'
       write (*,*) '  |_______ X'
       write (*,*) ' '
    end if

    ! cleanup
    deallocate(buff)
    deallocate(sbuff, level_rank)

  end subroutine ghex_cart_print_rank_topology

  subroutine print_cube(comm, dims, id, buff, order)
    integer(kind=4), intent(in) :: comm
    integer(kind=4), intent(in) :: dims(3), id, order
    integer(kind=4), intent(in), dimension(:,:), allocatable :: buff
    
    integer(kind=4) :: ierr, k, j, i, kk, n
    integer(kind=4) :: comm_size
    character(len=20) :: fmt, fmti

    call MPI_Comm_size(comm, comm_size, ierr)

    if(comm_size < 1000) then
       fmti="($' ',I3)"
    else
       fmti="($' ',I4)"
    endif
    do k=dims(3)-1,0,-1
       do j=dims(2)-1,0,-1
          fmt="(A1)"
          do kk=0,(j-1)*2+5
             write (*,fmt,ADVANCE='NO') " "
          end do
          fmt=fmti
          do i=0,dims(1)-1
             call ghex_cart_coord2rank(comm, dims, (/.false., .false., .false./), (/i, j, k/), n, order)
             write (*,fmt,ADVANCE='NO') buff(id,n)
          end do
          write (*,"(A1)",ADVANCE='NO') new_line(" ")
       end do
       write (*,"(A1)",ADVANCE='NO') new_line(" ")
    end do
  end subroutine print_cube
  
  subroutine split_type_to_name(split_type, name)
    integer(kind=4), intent(in) :: split_type
    character(len=32), intent(out) :: name
    
    select case (split_type)
    case (MPI_COMM_TYPE_SHARED)
       name = 'MPI_COMM_TYPE_SHARED'
    case (OMPI_COMM_TYPE_HWTHREAD)
       name = 'OMPI_COMM_TYPE_HWTHREAD'
    case (OMPI_COMM_TYPE_CORE)
       name = 'OMPI_COMM_TYPE_CORE'
    case (OMPI_COMM_TYPE_L1CACHE)
       name = 'OMPI_COMM_TYPE_L1CACHE'
    case (OMPI_COMM_TYPE_L2CACHE)
       name = 'OMPI_COMM_TYPE_L2CACHE'
    case (OMPI_COMM_TYPE_L3CACHE)
       name = 'OMPI_COMM_TYPE_L3CACHE'
    case (OMPI_COMM_TYPE_SOCKET)
       name = 'OMPI_COMM_TYPE_SOCKET'
    case (OMPI_COMM_TYPE_NUMA)
       name = 'OMPI_COMM_TYPE_NUMA'
    case (OMPI_COMM_TYPE_BOARD)
       name = 'OMPI_COMM_TYPE_BOARD'
    case (OMPI_COMM_TYPE_HOST)
       name = 'OMPI_COMM_TYPE_HOST'
    case (OMPI_COMM_TYPE_CU)
       name = 'OMPI_COMM_TYPE_CU'
    case (OMPI_COMM_TYPE_CLUSTER)
       name = 'OMPI_COMM_TYPE_CLUSTER'
    end select
  end subroutine split_type_to_name

END MODULE ghex_utils
