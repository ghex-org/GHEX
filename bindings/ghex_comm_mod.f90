MODULE ghex_comm_mod
  use iso_c_binding
  use ghex_message_mod
  use ghex_future_mod

  implicit none

  type, bind(c) :: ghex_communicator
     type(c_ptr) :: comm = c_null_ptr
  end type ghex_communicator
  
  interface
     
     ! callback type
     subroutine f_callback (rank, tag, mesg) bind(c)
       use iso_c_binding
       import ghex_shared_message       
       integer(c_int), value :: rank, tag
       type(ghex_shared_message), value :: mesg
     end subroutine f_callback

     type(ghex_communicator) function comm_new() bind(c)
       use iso_c_binding
       import ghex_communicator
     end function comm_new

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

     subroutine comm_send_cb_wrapped(comm, message, rank, tag, cb) bind(c, name="comm_send_cb")
       use iso_c_binding
       import ghex_communicator, ghex_shared_message
       type(ghex_communicator), value :: comm
       type(ghex_shared_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
     end subroutine comm_send_cb_wrapped

     type(ghex_future) function comm_send_wrapped(comm, message, rank, tag) bind(c, name="comm_send")
       use iso_c_binding
       import ghex_communicator, ghex_shared_message, ghex_future
       type(ghex_communicator), value :: comm
       type(ghex_shared_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
     end function comm_send_wrapped

     subroutine comm_send_multi_wrapped(comm, message, ranks, nranks, tag, cb) bind(c, name="comm_send_multi")
       use iso_c_binding
       import ghex_communicator, ghex_shared_message
       type(ghex_communicator), value :: comm
       type(ghex_shared_message), value :: message
       type(c_ptr), value :: ranks
       integer(c_int), value :: nranks
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
     end subroutine comm_send_multi_wrapped

     subroutine comm_recv_cb_wrapped(comm, message, rank, tag, cb) bind(c, name="comm_recv_cb")
       use iso_c_binding
       import ghex_communicator, ghex_shared_message
       type(ghex_communicator), value :: comm
       type(ghex_shared_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
       type(c_funptr), value :: cb
     end subroutine comm_recv_cb_wrapped
     
     type(ghex_future) function comm_recv_wrapped(comm, message, rank, tag) bind(c, name="comm_recv")
       use iso_c_binding
       import ghex_communicator, ghex_shared_message, ghex_future
       type(ghex_communicator), value :: comm
       type(ghex_shared_message), value :: message
       integer(c_int), value :: rank
       integer(c_int), value :: tag
     end function comm_recv_wrapped
     
     type(ghex_future) function comm_detach(comm, rank, tag) bind(c) 
       use iso_c_binding
       import ghex_communicator, ghex_future
       type(ghex_communicator), value :: comm
       integer(c_int), value :: rank
       integer(c_int), value :: tag
     end function comm_detach

  end interface

CONTAINS

  ! Need the wrappers for send/recv to enforce correct callback type.
  ! However, this results in (**f_callback) being passed to C
  ! instead of (*f_callback), at least with gfortran.

  ! TODO: check other compilers.
  
  subroutine comm_send_cb(comm, message, rank, tag, cbarg)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_shared_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), optional, pointer :: cbarg
    procedure(f_callback), pointer :: cb
    
    if (present(cbarg)) then
      cb => cbarg
    else
      cb => null()
    end if

    call comm_send_cb_wrapped(comm, message, rank, tag, c_funloc(cb))
  end subroutine comm_send_cb

  type(ghex_future) function comm_send(comm, message, rank, tag)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_shared_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    
    comm_send = comm_send_wrapped(comm, message, rank, tag)
  end function comm_send
  
  subroutine comm_send_multi(comm, message, ranks, tag, cbarg)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_shared_message), value :: message
    integer, dimension(:), target, intent(in) :: ranks
    integer, intent(in) :: tag
    procedure(f_callback), optional, pointer :: cbarg
    procedure(f_callback), pointer :: cb

    if (present(cbarg)) then
      cb => cbarg
    else
      cb => null()
    end if
    
    call comm_send_multi_wrapped(comm, message, c_loc(ranks), size(ranks), tag, c_funloc(cb))
  end subroutine comm_send_multi

  subroutine comm_recv_cb(comm, message, rank, tag, cb)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_shared_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    procedure(f_callback), pointer :: cb
    
    call comm_recv_cb_wrapped(comm, message, rank, tag, c_funloc(cb))
  end subroutine comm_recv_cb

  type(ghex_future) function comm_recv(comm, message, rank, tag)
    use iso_c_binding
    type(ghex_communicator), intent(in) :: comm
    type(ghex_shared_message), value :: message
    integer, intent(in) :: rank
    integer, intent(in) :: tag
    
    comm_recv = comm_recv_wrapped(comm, message, rank, tag)
  end function comm_recv

END MODULE ghex_comm_mod
