MODULE ghex_comm_mod
  use iso_c_binding
  use ghex_message_mod
  use ghex_request_mod
  use ghex_future_mod

  implicit none

  ! Communicator can return either a future, or a request handle to track the progress.
  ! requests are returned by the callback-based API
  ! futures are returned by the 
  
  ! ---------------------
  ! --- module types
  ! ---------------------
  type, bind(c) :: ghex_communicator
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_communicator

  type, bind(c) :: ghex_progress_status
     integer(c_int) :: num_sends = 0
     integer(c_int) :: num_recvs = 0
     integer(c_int) :: num_cancels = 0
  end type ghex_progress_status


  ! ---------------------
  ! --- module C interfaces
  ! ---------------------
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

     type(ghex_progress_status) function ghex_comm_progress(comm) bind(c)
       use iso_c_binding
       import ghex_communicator, ghex_progress_status
       type(ghex_communicator), value :: comm
     end function ghex_comm_progress

     subroutine ghex_comm_barrier(comm) bind(c)
       use iso_c_binding
       import ghex_communicator
       type(ghex_communicator), value :: comm
     end subroutine ghex_comm_barrier

     ! WRAPPED - you should call ghex_comm_send_cb
     ! send a message with callback:
     ! message is taken over by ghex, and users copy is freed
     ! callback is called upon completion of the communication request
     ! request (optional) is returned and can be tested for completion, but not waited on
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

     ! WRAPPED - you should call ghex_comm_post_send_cb
     ! post a send on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! callback is called upon completion of the communication request
     ! request (optional) is returned and can be tested for completion, but not waited on
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

     ! post a send on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! future has to be tested / waited on to assure completion
     subroutine ghex_comm_post_send(comm, message, rank, tag, future) bind(c, name="ghex_comm_post_send")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_future
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(ghex_future) :: future
     end subroutine ghex_comm_post_send

     ! WRAPPED - you should call ghex_comm_post_send_multi
     ! post a send to MULTIPLE destinations on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! future has to be tested / waited on to assure completion
     subroutine ghex_comm_post_send_multi_wrapped(comm, message, ranks, nranks, tag, future) bind(c, name="ghex_comm_post_send_multi")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_future_multi
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       type(c_ptr), value :: ranks
       integer(c_int), value :: nranks
       integer(c_int), value :: tag
       type(ghex_future_multi) :: future
     end subroutine ghex_comm_post_send_multi_wrapped

     ! WRAPPED - you should call ghex_comm_recv_cb_wrapped
     ! recv a message with callback:
     ! message is taken over by ghex, and users copy is freed
     ! callback is called upon completion of the communication request
     ! request (optional) is returned and can be tested for completion, but not waited on
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

     ! WRAPPED - you should call ghex_comm_post_recv_cb
     ! post a recv on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! callback is called upon completion of the communication request
     ! request (optional) is returned and can be tested for completion, but not waited on
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

     ! post a recv on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! future has to be tested / waited on to assure completion
     subroutine ghex_comm_post_recv(comm, message, rank, tag, future) bind(c, name="ghex_comm_post_recv")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_future
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(ghex_future) :: future
     end subroutine ghex_comm_post_recv

     ! WRAPPED - you should call ghex_comm_resubmit_recv
     ! resubmit a recv on a message inside a completion callback:
     ! callback is called upon completion of the communication request
     ! request (optional) is returned and can be tested for completion, but not waited on
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


  ! ---------------------
  ! --- generic ghex interfaces
  ! ---------------------
  interface ghex_free
     subroutine ghex_comm_free(comm) bind(c, name="ghex_obj_free")
       use iso_c_binding
       import ghex_communicator
       type(ghex_communicator) :: comm
     end subroutine ghex_comm_free
  end interface ghex_free

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

  subroutine ghex_comm_post_send_multi(comm, message, ranks, tag, future)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, dimension(:), intent(in), target :: ranks
    integer, intent(in) :: tag
    type(ghex_future_multi) :: future
    call ghex_comm_post_send_multi_wrapped(comm, message, c_loc(ranks), size(ranks), tag, future)
  end subroutine ghex_comm_post_send_multi

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
