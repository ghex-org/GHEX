MODULE ghex_message_mod
  use iso_c_binding
  implicit none

  integer, public, parameter :: ALLOCATOR_STD             = 1
  integer, public, parameter :: ALLOCATOR_PERSISTENT_STD  = 2
  integer, public, parameter :: ALLOCATOR_HOST            = 3
  integer, public, parameter :: ALLOCATOR_PERSISTENT_HOST = 4
  integer, public, parameter :: ALLOCATOR_GPU             = 5
  integer, public, parameter :: ALLOCATOR_PERSISTENT_GPU  = 6

  type, bind(c) :: ghex_message
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_message
  
  interface
     type(ghex_message) function message_new(size, allocator) bind(c)
       use iso_c_binding
       import ghex_message
       integer(c_size_t), value :: size
       integer(c_int), value :: allocator
     end function message_new

     subroutine message_delete(message) bind(c)
       use iso_c_binding
       import ghex_message
       ! reference, not a value - fortran variable is reset to null 
       type(ghex_message) :: message
     end subroutine message_delete

     integer(c_int) function message_use_count(message) bind(c)
       use iso_c_binding
       import ghex_message
       type(ghex_message), value :: message
     end function message_use_count

     logical function message_is_host(message) bind(c)
       use iso_c_binding
       import ghex_message
       type(ghex_message), value :: message
     end function message_is_host

     type(c_ptr) function message_data_wrapped(message, capacity) bind(c, name='message_data')
       use iso_c_binding
       import ghex_message
       type(ghex_message), value :: message
       integer(c_size_t), intent(out) :: capacity
     end function message_data_wrapped

     subroutine message_resize(message, size) bind(c)
       use iso_c_binding
       import ghex_message
       type(ghex_message), value :: message
       integer(c_size_t), value :: size
     end subroutine message_resize
  end interface

CONTAINS

  function message_data(message)
    use iso_c_binding
    type(ghex_message), value :: message
    integer(1), dimension(:), pointer :: message_data

    type(c_ptr) :: c_data = c_null_ptr
    integer(c_size_t) :: capacity

    ! get the data pointer
    c_data = message_data_wrapped(message, capacity)
    call c_f_pointer(c_data, message_data, [capacity])
  end function message_data

END MODULE ghex_message_mod
