MODULE ghex_comm_mod
  use iso_c_binding
  use ghex_message_mod
  use ghex_future_mod
  use ghex_request_mod

  implicit none

  type, bind(c) :: ghex_communicator
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_communicator

  interface

     ! callback type
     subroutine f_callback (message, rank, tag) bind(c)
       use iso_c_binding
       import ghex_message
       type(ghex_message), value :: message
       integer(c_int), value :: rank, tag
     end subroutine f_callback

     subroutine comm_delete(comm) bind(c)
       use iso_c_binding
       import ghex_communicator
       ! reference, not a value - fortran variable is reset to null
       type(ghex_communicator) :: comm
     end subroutine comm_delete

     integer(c_int) function comm_progress(comm) bind(c)
       use iso_c_binding
       import ghex_communicator
       type(ghex_communicator), value :: comm
     end function comm_progress

     type(ghex_request) function comm_send_cb_wrapped(comm, message, rank, tag, cb) bind(c, name="comm_send_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
     end function comm_send_cb_wrapped

     type(ghex_future) function comm_send_wrapped(comm, message, rank, tag) bind(c, name="comm_send")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_future
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
     end function comm_send_wrapped

     type(ghex_request) function comm_recv_cb_wrapped(comm, message, rank, tag, cb) bind(c, name="comm_recv_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
     end function comm_recv_cb_wrapped

     type(ghex_future) function comm_recv_wrapped(comm, message, rank, tag) bind(c, name="comm_recv")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_future
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
     end function comm_recv_wrapped

  end interface

CONTAINS

  ! Need the wrappers for send/recv to enforce correct callback type.
  ! However, this results in (**f_callback) being passed to C
  ! instead of (*f_callback), at least with gfortran.

  ! TODO: check other compilers.

  type(ghex_request) function comm_send_cb(comm, message, rank, tag, cbarg)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), optional, pointer :: cbarg
    procedure(f_callback), pointer :: cb

    if (present(cbarg)) then
       cb => cbarg
    else
       cb => null()
    end if

    comm_send_cb = comm_send_cb_wrapped(comm, message, rank, tag, c_funloc(cb))
  end function comm_send_cb

  type(ghex_future) function comm_send(comm, message, rank, tag)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag

    comm_send = comm_send_wrapped(comm, message, rank, tag)
  end function comm_send

  ! subroutine comm_send_multi(comm, message, ranks, tag, cbarg)
  !   use iso_c_binding
  !   type(ghex_communicator), intent(in) :: comm
  !   type(ghex_message), value :: message
  !   integer, dimension(:), target, intent(in) :: ranks
  !   integer, intent(in) :: tag
  !   procedure(f_callback), optional, pointer :: cbarg
  !   procedure(f_callback), pointer :: cb

  !   if (present(cbarg)) then
  !     cb => cbarg
  !   else
  !     cb => null()
  !   end if

  !   call comm_send_multi_wrapped(comm, message, c_loc(ranks), size(ranks), tag, c_funloc(cb))
  ! end subroutine comm_send_multi

  type(ghex_request) function comm_recv_cb(comm, message, rank, tag, cbarg)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), pointer :: cbarg
    procedure(f_callback), pointer :: cb
    cb => cbarg
    comm_recv_cb = comm_recv_cb_wrapped(comm, message, rank, tag, c_funloc(cb))
  end function comm_recv_cb

  type(ghex_future) function comm_recv(comm, message, rank, tag)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag

    comm_recv = comm_recv_wrapped(comm, message, rank, tag)
  end function comm_recv

END MODULE ghex_comm_mod
