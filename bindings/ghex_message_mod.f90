MODULE ghex_message_mod
  use iso_c_binding
  implicit none

  integer, public, parameter :: ALLOCATOR_STD             = 1
  integer, public, parameter :: ALLOCATOR_PERSISTENT_STD  = 2
  integer, public, parameter :: ALLOCATOR_HOST            = 3
  integer, public, parameter :: ALLOCATOR_PERSISTENT_HOST = 4
  integer, public, parameter :: ALLOCATOR_GPU             = 5
  integer, public, parameter :: ALLOCATOR_PERSISTENT_GPU  = 6

  type, bind(c) :: ghex_shared_message
     type(c_ptr) :: msg = c_null_ptr
  end type ghex_shared_message
  
  interface
     type(ghex_shared_message) function shared_message_new(size, allocator) bind(c)
       use iso_c_binding
       import ghex_shared_message
       integer(c_size_t), value :: size
       integer(c_int), value :: allocator
     end function shared_message_new

     subroutine shared_message_delete(shared_message) bind(c)
       use iso_c_binding
       import ghex_shared_message
       ! reference, not a value - fortran variable is reset to null 
       type(ghex_shared_message) :: shared_message
     end subroutine shared_message_delete

     integer(c_int) function shared_message_use_count(shared_message) bind(c)
       use iso_c_binding
       import ghex_shared_message
       type(ghex_shared_message), value :: shared_message
     end function shared_message_use_count

     logical function shared_message_is_host(shared_message) bind(c)
       use iso_c_binding
       import ghex_shared_message
       type(ghex_shared_message), value :: shared_message
     end function shared_message_is_host

     type(c_ptr) function shared_message_data_wrapped(shared_message, capacity) bind(c, name='shared_message_data')
       use iso_c_binding
       import ghex_shared_message
       type(ghex_shared_message), value :: shared_message
       integer(c_size_t), intent(out) :: capacity
     end function shared_message_data_wrapped

     subroutine shared_message_resize(shared_message, size) bind(c)
       use iso_c_binding
       import ghex_shared_message
       type(ghex_shared_message), value :: shared_message
       integer(c_size_t), value :: size
     end subroutine shared_message_resize
  end interface

CONTAINS

  function shared_message_data(shared_message)
    use iso_c_binding
    type(ghex_shared_message), value :: shared_message
    integer(1), dimension(:), pointer :: shared_message_data

    type(c_ptr) :: c_data = c_null_ptr
    integer(c_size_t) :: capacity

    ! get the data pointer
    c_data = shared_message_data_wrapped(shared_message, capacity)
    call c_f_pointer(c_data, shared_message_data, [capacity])
  end function shared_message_data

END MODULE ghex_message_mod
