MODULE ghex_utils
  use mpi
  use ghex_defs

#define OPEN_MPI
  implicit none

  interface

     integer(c_int) function ghex_get_current_cpu() bind(c)
       use iso_c_binding
     end function ghex_get_current_cpu

  end interface
contains

  ! obtain compute node ID of the calling rank
  subroutine ghex_get_noderank(comm, noderank)
    implicit none
    integer(kind=4), intent(in)  :: comm
    integer(kind=4), intent(out) :: noderank
    integer(kind=4), dimension(:), allocatable :: sbuff, rbuff
    integer(kind=4) :: nodeid, rank, size, ierr, ii
    integer(kind=4) :: shmcomm

    ! old rank
    call MPI_Comm_rank(comm, rank, ierr)
    call MPI_Comm_size(comm, size, ierr)

    ! communication buffers
    allocate(sbuff(1:size), rbuff(1:size), Source=0)

    ! create local communicator
#ifdef OPEN_MPI
    call MPI_Comm_split_type(comm, OMPI_COMM_TYPE_NUMA, 0, MPI_INFO_NULL, shmcomm, ierr)
#else
    call MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, shmcomm, ierr)
#endif

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


  ! Renumber the MPI ranks so that node-local ranks form
  ! compact boxes. Assuming the compute nodes are arranged
  ! into nodedims(3) dimensions, the per-compute node domain
  ! grid will be dims/nodedims.
  !
  ! A new communicator is created with this rank numbering.
  ! It can then be used as usual with rank2coord, coord2rank,
  ! and create_subcomm.
  !
  ! TODO : find a good automatic way to compute nodedims: split the grid into compact sub-grids,
  ! which fit into a single compute node. Or use Z-curves instead.
  subroutine ghex_cart_remap_ranks(comm, dims, nodedims, newcomm, iorder)
    implicit none
    integer(kind=4), intent(in)  :: comm, dims(3), nodedims(3)
    integer(kind=4), intent(out) :: newcomm
    integer(kind=4) ::          newrank, newcoord(3)
    integer(kind=4) :: shmcomm, shmrank, shmcoord(3), shmdims(3), shmsize, rank, size
    integer(kind=4) ::         noderank, nodecoord(3)
    integer(kind=4) :: ierr, order
    character(len=20) :: fmti
    integer(kind=4), optional :: iorder

    if (present(iorder)) then
       order = iorder
    else
       order = CartOrderDefault
    endif    

    ! total number of ranks
    call MPI_Comm_rank(comm, rank, ierr)
    call MPI_Comm_size(comm, size, ierr)

    ! node-local communicator and node-local rank
#ifdef OPEN_MPI
    call MPI_Comm_split_type(comm, OMPI_COMM_TYPE_NUMA, 0, MPI_INFO_NULL, shmcomm, ierr)
#else
    call MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, shmcomm, ierr)
#endif
    call MPI_Comm_rank(shmcomm, shmrank, ierr)
    call MPI_Comm_size(shmcomm, shmsize, ierr)
    call MPI_Comm_Disconnect(shmcomm, ierr)

    if (shmsize*product(nodedims) /= size) then
       if (rank == 0) then
          if (size>1000) then
             fmti="(A,I5,A,I5,A,I5)"
          else
             fmti="(A,I3,A,I3,A,I3)"
          end if
          write (*,*) ' ERROR: Wrong node space dimensions: number of ranks doesnt match.'
          write (*,fmti) '  There is ', product(nodedims), ' nodes and ', shmsize, ' ranks per node, but the total number of ranks is ', size
       end if
       call MPI_Finalize(ierr)
       call exit
    end if

    ! which compute node are we on
    call ghex_get_noderank(comm, noderank)

    ! cartesian node coordinates in the node space
    call ghex_cart_rank2coord(MPI_COMM_NULL, nodedims, noderank, nodecoord, order)

    ! cartesian shmrank coordinates in node-local rank space
    shmdims = dims/nodedims
    call ghex_cart_rank2coord(MPI_COMM_NULL, shmdims, shmrank, shmcoord, order)

    ! new rank coordinates in remapped global rank space
    newcoord = nodecoord*shmdims + shmcoord
    call ghex_cart_coord2rank(MPI_COMM_NULL, dims, (/.false., .false., .false./), newcoord, newrank, order)

    ! create the new communicator with remapped ranks
    call MPI_Comm_split(comm, 0, newrank, newcomm, ierr)

  end subroutine ghex_cart_remap_ranks

  subroutine ghex_print_rank2node(comm)
    integer(kind=4), intent(in) :: comm
    integer(kind=4), dimension(:,:), allocatable :: buff
    integer(kind=4) :: sbuff(3), ierr, i
    integer(kind=4) :: rank, size, noderank, orank

    call ghex_get_noderank(comm, noderank)

    ! obtain all values at master
    call MPI_Comm_rank(mpi_comm_world, orank, ierr)
    call MPI_Comm_rank(comm, rank, ierr)
    call MPI_Comm_size(comm, size, ierr)

    allocate(buff(1:3,0:size-1), Source=0)
    sbuff(1) = noderank
    sbuff(2) = rank
    sbuff(3) = orank
    call MPI_Gather(sbuff, 3, MPI_INT, buff, 3, MPI_INT, 0, comm, ierr)

    if (rank==0) then       
       do i=0,size-1
          write (*,"(I3)",ADVANCE='NO') buff(1,i)
       end do
       write (*,*)
       do i=0,size-1
          write (*,"(I3)",ADVANCE='NO') buff(2,i)
       end do
       write (*,*)
       do i=0,size-1
          write (*,"(I3)",ADVANCE='NO') buff(3,i)
       end do
       write (*,*)
    end if

    deallocate(buff)    
  end subroutine ghex_print_rank2node

  subroutine ghex_cart_print_rank2node(comm, dims, iorder)
    implicit none
    integer(kind=4), intent(in) :: comm, dims(3)
    integer(kind=4), dimension(:,:), allocatable :: buff
    integer(kind=4) :: sbuff(4), ierr, k, j, i, kk, n
    integer(kind=4) :: rank, size, noderank, order, orank
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
    sbuff(1) = noderank
    sbuff(2) = rank
    sbuff(3) = orank
    sbuff(4) = ghex_get_current_cpu()
    if (sbuff(4) >= 36) then
       sbuff(4) = sbuff(4) - 36
    end if
    call MPI_Gather(sbuff, 4, MPI_INT, buff, 4, MPI_INT, 0, comm, ierr)

    if (rank==0) then
       write (*,*) ' '
       write (*,*) ' Rank to node mapping '
       write (*,*) ' '

       if(size < 1000) then
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
                write (*,fmt,ADVANCE='NO') buff(1,n)
             end do
             write (*,"(A1)",ADVANCE='NO') new_line(" ")
          end do
          write (*,"(A1)",ADVANCE='NO') new_line(" ")
       end do

       write (*,*) ' '
       write (*,*) ' Rank layout '
       write (*,*) ' '

       if(size < 1000) then
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
                write (*,fmt,ADVANCE='NO') buff(2,n)
             end do
             write (*,"(A1)",ADVANCE='NO') new_line(" ")
          end do
          write (*,"(A1)",ADVANCE='NO') new_line(" ")
       end do

       write (*,*) ' '
       write (*,*) ' MPI_COMM_WORLD layout '
       write (*,*) ' '

       if(size < 1000) then
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
                write (*,fmt,ADVANCE='NO') buff(3,n)
             end do
             write (*,"(A1)",ADVANCE='NO') new_line(" ")
          end do
          write (*,"(A1)",ADVANCE='NO') new_line(" ")
       end do

       write (*,*) ' '
       write (*,*) ' CPU '
       write (*,*) ' '

       if(size < 1000) then
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
                write (*,fmt,ADVANCE='NO') buff(4,n)
             end do
             write (*,"(A1)",ADVANCE='NO') new_line(" ")
          end do
          write (*,"(A1)",ADVANCE='NO') new_line(" ")
       end do

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

END MODULE ghex_utils
