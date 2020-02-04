MODULE ghex_comm_mod
  use iso_c_binding
  use ghex_message_mod
  use ghex_request_mod
  use ghex_future_mod

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

     type(ghex_communicator) function ghex_comm_new() bind(c)
       use iso_c_binding
       import ghex_communicator
     end function ghex_comm_new

     subroutine ghex_comm_delete(comm) bind(c)
       use iso_c_binding
       import ghex_communicator
       ! reference, not a value - fortran variable is reset to null
       type(ghex_communicator) :: comm
     end subroutine ghex_comm_delete

     integer(c_int) function ghex_comm_rank(comm) bind(c)
       use iso_c_binding
       import ghex_communicator
       type(ghex_communicator), value :: comm
     end function ghex_comm_rank

     integer(c_int) function ghex_comm_size(comm) bind(c)
       use iso_c_binding
       import ghex_communicator
       type(ghex_communicator), value :: comm
     end function ghex_comm_size

     integer(c_int) function ghex_comm_progress(comm) bind(c)
       use iso_c_binding
       import ghex_communicator
       type(ghex_communicator), value :: comm
     end function ghex_comm_progress

     subroutine ghex_comm_barrier(comm) bind(c)
       use iso_c_binding
       import ghex_communicator
       type(ghex_communicator), value :: comm
     end subroutine ghex_comm_barrier

     subroutine ghex_comm_send_cb_wrapped(comm, message, rank, tag, cb, req) bind(c, name="ghex_comm_send_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request
       type(ghex_communicator), value :: comm
       type(ghex_message) :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request), optional :: req
     end subroutine ghex_comm_send_cb_wrapped

     subroutine ghex_comm_post_send_cb_wrapped(comm, message, rank, tag, cb, req) bind(c, name="ghex_comm_post_send_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request), optional :: req
     end subroutine ghex_comm_post_send_cb_wrapped

     subroutine ghex_comm_post_send(comm, message, rank, tag, future) bind(c, name="ghex_comm_post_send")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_future
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(ghex_future) :: future
     end subroutine ghex_comm_post_send

     subroutine ghex_comm_recv_cb_wrapped(comm, message, rank, tag, cb, req) bind(c, name="ghex_comm_recv_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request
       type(ghex_communicator), value :: comm
       type(ghex_message) :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request), optional :: req
     end subroutine ghex_comm_recv_cb_wrapped

     subroutine ghex_comm_post_recv_cb_wrapped(comm, message, rank, tag, cb, req) bind(c, name="ghex_comm_post_recv_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request), optional :: req
     end subroutine ghex_comm_post_recv_cb_wrapped

     subroutine ghex_comm_post_recv(comm, message, rank, tag, future) bind(c, name="ghex_comm_post_recv")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_future
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(ghex_future) :: future
     end subroutine ghex_comm_post_recv

     subroutine ghex_comm_resubmit_recv_wrapped(comm, message, rank, tag, cb, req) bind(c, name="ghex_comm_resubmit_recv")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request), optional :: req
     end subroutine ghex_comm_resubmit_recv_wrapped

  end interface

CONTAINS

  ! Need the wrappers for send/recv to enforce correct callback type.
  subroutine ghex_comm_send_cb(comm, message, rank, tag, cbarg, req)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message) :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), optional, pointer :: cbarg
    type(ghex_request), optional :: req
    procedure(f_callback), pointer :: cb

    if (present(cbarg)) then
       cb => cbarg
    else
       cb => null()
    end if

    call ghex_comm_send_cb_wrapped(comm, message, rank, tag, c_funloc(cb), req)
  end subroutine ghex_comm_send_cb

  subroutine ghex_comm_post_send_cb(comm, message, rank, tag, cbarg, req)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), optional, pointer :: cbarg
    type(ghex_request), optional :: req
    procedure(f_callback), pointer :: cb

    if (present(cbarg)) then
       cb => cbarg
    else
       cb => null()
    end if

    call ghex_comm_post_send_cb_wrapped(comm, message, rank, tag, c_funloc(cb), req)
  end subroutine ghex_comm_post_send_cb

  subroutine ghex_comm_recv_cb(comm, message, rank, tag, cbarg, req)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message) :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), pointer :: cbarg
    type(ghex_request), optional :: req
    procedure(f_callback), pointer :: cb

    cb => cbarg
    call ghex_comm_recv_cb_wrapped(comm, message, rank, tag, c_funloc(cb), req)
  end subroutine ghex_comm_recv_cb

  subroutine ghex_comm_post_recv_cb(comm, message, rank, tag, cbarg, req)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), pointer :: cbarg
    type(ghex_request), optional :: req
    procedure(f_callback), pointer :: cb

    cb => cbarg
    call ghex_comm_post_recv_cb_wrapped(comm, message, rank, tag, c_funloc(cb), req)
  end subroutine ghex_comm_post_recv_cb

  subroutine ghex_comm_resubmit_recv(comm, message, rank, tag, cbarg, req)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), pointer :: cbarg
    type(ghex_request), optional :: req
    procedure(f_callback), pointer :: cb

    cb => cbarg
    call ghex_comm_resubmit_recv_wrapped(comm, message, rank, tag, c_funloc(cb), req)
  end subroutine ghex_comm_resubmit_recv

END MODULE ghex_comm_mod
