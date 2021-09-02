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

  type, bind(c) :: ghex_cb_user_data
     type(c_ptr) :: data
  end type ghex_cb_user_data

  ! ---------------------
  ! --- module C interfaces
  ! ---------------------
  interface

     ! callback type
     subroutine f_callback (message, rank, tag, user_data) bind(c)
       use iso_c_binding
       import ghex_message, ghex_cb_user_data
       type(ghex_message), value :: message
       integer(c_int), value :: rank, tag
       type(ghex_cb_user_data), value :: user_data
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

     subroutine ghex_comm_barrier(comm, type) bind(c)
       use iso_c_binding
       import ghex_communicator
       type(ghex_communicator), value :: comm
       integer, value:: type
     end subroutine ghex_comm_barrier


     ! -----------------------------------------------------------------------------------------
     ! SEND requests
     ! -----------------------------------------------------------------------------------------
     
     ! post a send on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! future has to be tested / waited on to assure completion
     subroutine ghex_comm_post_send(comm, message, rank, tag, future) \
       bind(c, name="ghex_comm_post_send")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_future
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(ghex_future) :: future
     end subroutine ghex_comm_post_send

     ! WRAPPED - you should call ghex_comm_post_send_cb
     ! post a send on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! callback is called upon completion of the communication request
     ! request (optional) is returned and can be tested for completion, but not waited on
     subroutine ghex_comm_post_send_cb_wrapped(comm, message, rank, tag, cb, req, user_data) \
       bind(c, name="ghex_comm_post_send_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request, ghex_cb_user_data
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request) :: req
       type(ghex_cb_user_data), value :: user_data
     end subroutine ghex_comm_post_send_cb_wrapped
   
     ! WRAPPED - you should call ghex_comm_send_cb
     ! send a message with callback:
     ! message is taken over by ghex, and users copy is freed
     ! callback is called upon completion of the communication request
     ! request (optional) is returned and can be tested for completion, but not waited on
     subroutine ghex_comm_send_cb_wrapped(comm, message, rank, tag, cb, req, user_data) \
       bind(c, name="ghex_comm_send_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request, ghex_cb_user_data
       type(ghex_communicator), value :: comm
       type(ghex_message) :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request) :: req
       type(ghex_cb_user_data), value :: user_data
     end subroutine ghex_comm_send_cb_wrapped

     ! -----------------------------------------------------------------------------------------
     ! SEND_MULTI requests
     ! -----------------------------------------------------------------------------------------
     
     ! WRAPPED - you should call ghex_comm_post_send_multi
     ! post a send to MULTIPLE destinations on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! future has to be tested / waited on to assure completion
     subroutine ghex_comm_post_send_multi_wrapped(comm, message, ranks, nranks, tag, future) \
       bind(c, name="ghex_comm_post_send_multi")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_future_multi
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       type(c_ptr), value :: ranks
       integer(c_int), value :: nranks
       integer(c_int), value :: tag
       type(ghex_future_multi) :: future
     end subroutine ghex_comm_post_send_multi_wrapped

     ! WRAPPED - you should call ghex_comm_post_send_multi_cb
     ! post a send to MULTIPLE destinations on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! future has to be tested / waited on to assure completion
     subroutine ghex_comm_post_send_multi_cb_wrapped(comm, message, ranks, nranks, tag, cb, req, user_data) \
       bind(c, name="ghex_comm_post_send_multi_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request_multi, ghex_cb_user_data
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       type(c_ptr), value :: ranks
       integer(c_int), value :: nranks
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request_multi) :: req
       type(ghex_cb_user_data), value :: user_data
     end subroutine ghex_comm_post_send_multi_cb_wrapped

     ! WRAPPED - you should call ghex_comm_post_send_multi_cb
     ! post a send to MULTIPLE destinations on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! future has to be tested / waited on to assure completion
     subroutine ghex_comm_send_multi_cb_wrapped(comm, message, ranks, nranks, tag, cb, req, user_data) \
       bind(c, name="ghex_comm_send_multi_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request_multi, ghex_cb_user_data
       type(ghex_communicator), value :: comm
       type(ghex_message) :: message
       type(c_ptr), value :: ranks
       integer(c_int), value :: nranks
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request_multi) :: req
       type(ghex_cb_user_data), value :: user_data
     end subroutine ghex_comm_send_multi_cb_wrapped

     
     ! -----------------------------------------------------------------------------------------
     ! RECV requests
     ! -----------------------------------------------------------------------------------------
     
     ! post a recv on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! future has to be tested / waited on to assure completion
     subroutine ghex_comm_post_recv(comm, message, rank, tag, future) \
       bind(c, name="ghex_comm_post_recv")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_future
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(ghex_future) :: future
     end subroutine ghex_comm_post_recv

     ! WRAPPED - you should call ghex_comm_post_recv_cb
     ! post a recv on a message: message is still owned by the user
     ! and has to be freed when necessary
     ! callback is called upon completion of the communication request
     ! request (optional) is returned and can be tested for completion, but not waited on
     subroutine ghex_comm_post_recv_cb_wrapped(comm, message, rank, tag, cb, req, user_data) \
       bind(c, name="ghex_comm_post_recv_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request, ghex_cb_user_data
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request) :: req
       type(ghex_cb_user_data), value :: user_data
     end subroutine ghex_comm_post_recv_cb_wrapped

     ! WRAPPED - you should call ghex_comm_recv_cb_wrapped
     ! recv a message with callback:
     ! message is taken over by ghex, and users copy is freed
     ! callback is called upon completion of the communication request
     ! request (optional) is returned and can be tested for completion, but not waited on
     subroutine ghex_comm_recv_cb_wrapped(comm, message, rank, tag, cb, req, user_data) \
       bind(c, name="ghex_comm_recv_cb")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request, ghex_cb_user_data
       type(ghex_communicator), value :: comm
       type(ghex_message) :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request) :: req
       type(ghex_cb_user_data), value :: user_data
     end subroutine ghex_comm_recv_cb_wrapped

     ! -----------------------------------------------------------------------------------------
     ! resubmission of recv requests from inside callbacks        
     ! -----------------------------------------------------------------------------------------

     ! WRAPPED - you should call ghex_comm_resubmit_recv
     ! resubmit a recv on a message inside a completion callback:
     ! callback is called upon completion of the communication request
     ! request (optional) is returned and can be tested for completion, but not waited on
     subroutine ghex_comm_resubmit_recv_wrapped(comm, message, rank, tag, cb, req, user_data) \
       bind(c, name="ghex_comm_resubmit_recv")
       use iso_c_binding
       import ghex_communicator, ghex_message, ghex_request, ghex_cb_user_data
       type(ghex_communicator), value :: comm
       type(ghex_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
       type(ghex_request) :: req
       type(ghex_cb_user_data), value :: user_data
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

  ! Need the wrappers for send/recv to enforce correct callback type,
  ! and to handle optional arguments

  subroutine ghex_comm_post_send_cb(comm, message, rank, tag, cb, req, user_data)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), optional, pointer :: cb
    type(ghex_request), optional :: req
    type(ghex_cb_user_data), optional :: user_data
    
    ! local variables
    procedure(f_callback), pointer :: lcb
    type(ghex_request) :: lreq
    type(ghex_cb_user_data) :: luser_data

    ! This is needed for GCC. Otherwise c_funloc(cart_nbor) doesn't work correctly
    ! This is a difference wrt. Intel compiler
    if (present(cb)) then
      lcb => cb
    else
      lcb => null()
    end if

    if (present(user_data)) then
      luser_data = user_data
    end if

    call ghex_comm_post_send_cb_wrapped(comm, message, rank, tag, c_funloc(lcb), lreq, luser_data)

    if (present(req)) then
      req = lreq
    end if
  end subroutine ghex_comm_post_send_cb
  
  subroutine ghex_comm_send_cb(comm, message, rank, tag, cb, req, user_data)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message) :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), optional, pointer :: cb
    type(ghex_request), optional :: req
    type(ghex_cb_user_data), optional :: user_data

    ! local variables
    procedure(f_callback), pointer :: lcb
    type(ghex_request) :: lreq
    type(ghex_cb_user_data) :: luser_data

    lcb => null()
    if (present(cb)) then
      lcb => cb
    end if

    if (present(user_data)) then
      luser_data = user_data
    end if

    call ghex_comm_send_cb_wrapped(comm, message, rank, tag, c_funloc(lcb), lreq, luser_data)

    if (present(req)) then
      req = lreq
    end if
  end subroutine ghex_comm_send_cb

  subroutine ghex_comm_post_send_multi(comm, message, ranks, tag, future)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, dimension(:), intent(in), target :: ranks
    integer, intent(in) :: tag
    type(ghex_future_multi) :: future
    
    call ghex_comm_post_send_multi_wrapped(comm, message, c_loc(ranks), size(ranks), tag, future)
  end subroutine ghex_comm_post_send_multi

  subroutine ghex_comm_post_send_multi_cb(comm, message, ranks, tag, cb, req, user_data)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, dimension(:), intent(in), target :: ranks
    integer, intent(in) :: tag
    procedure(f_callback), optional, pointer :: cb
    type(ghex_request_multi), optional :: req
    type(ghex_cb_user_data), optional :: user_data
    
    ! local variables
    procedure(f_callback), pointer :: lcb
    type(ghex_request_multi) :: lreq
    type(ghex_cb_user_data) :: luser_data

    if (present(cb)) then
      lcb => cb
    else
      lcb => null()
    end if

    if (present(user_data)) then
      luser_data = user_data
    end if
   
    call ghex_comm_post_send_multi_cb_wrapped(comm, message, c_loc(ranks), size(ranks), tag, c_funloc(lcb), lreq, luser_data)    

    if (present(req)) then
      req = lreq
    end if
  end subroutine ghex_comm_post_send_multi_cb

  subroutine ghex_comm_send_multi_cb(comm, message, ranks, tag, cb, req, user_data)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message) :: message
    integer, dimension(:), intent(in), target :: ranks
    integer, intent(in) :: tag
    procedure(f_callback), optional, pointer :: cb
    type(ghex_request_multi), optional :: req
    type(ghex_cb_user_data), optional :: user_data
    
    ! local variables
    procedure(f_callback), pointer :: lcb
    type(ghex_request_multi) :: lreq
    type(ghex_cb_user_data) :: luser_data
    
    if (present(cb)) then
      lcb => cb
    else
      lcb => null()
    end if

    if (present(user_data)) then
      luser_data = user_data
    end if
   
    call ghex_comm_send_multi_cb_wrapped(comm, message, c_loc(ranks), size(ranks), tag, c_funloc(lcb), lreq, luser_data)    

    if (present(req)) then
      req = lreq
    end if
  end subroutine ghex_comm_send_multi_cb

  subroutine ghex_comm_post_recv_cb(comm, message, rank, tag, cb, req, user_data)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), pointer :: cb
    type(ghex_request), optional :: req
    type(ghex_cb_user_data), optional :: user_data

    ! local variables
    procedure(f_callback), pointer :: lcb
    type(ghex_request) :: lreq
    type(ghex_cb_user_data) :: luser_data

    lcb => cb

    if (present(user_data)) then
      luser_data = user_data
    end if

    call ghex_comm_post_recv_cb_wrapped(comm, message, rank, tag, c_funloc(lcb), lreq, luser_data)

    if (present(req)) then
      req = lreq
    end if
  end subroutine ghex_comm_post_recv_cb

  subroutine ghex_comm_recv_cb(comm, message, rank, tag, cb, req, user_data)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message) :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), pointer :: cb
    type(ghex_request), optional :: req
    type(ghex_cb_user_data), optional :: user_data

    ! local variables
    procedure(f_callback), pointer :: lcb
    type(ghex_request) :: lreq
    type(ghex_cb_user_data) :: luser_data

    lcb => cb

    if (present(user_data)) then
      luser_data = user_data
    end if

    call ghex_comm_recv_cb_wrapped(comm, message, rank, tag, c_funloc(lcb), lreq, luser_data)

    if (present(req)) then
      req = lreq
    end if
  end subroutine ghex_comm_recv_cb

  subroutine ghex_comm_resubmit_recv(comm, message, rank, tag, cb, req, user_data)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), pointer :: cb
    type(ghex_request), optional :: req
    type(ghex_cb_user_data), optional :: user_data

    ! local variables
    procedure(f_callback), pointer :: lcb
    type(ghex_request) :: lreq
    type(ghex_cb_user_data) :: luser_data

    lcb => cb

    if (present(user_data)) then
      luser_data = user_data
    end if

    call ghex_comm_resubmit_recv_wrapped(comm, message, rank, tag, c_funloc(lcb), lreq, luser_data)

    if (present(req)) then
      req = lreq
    end if
  end subroutine ghex_comm_resubmit_recv

END MODULE ghex_comm_mod
