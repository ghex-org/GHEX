MODULE ghex_message_mod
  use iso_c_binding
  implicit none

  ! ---------------------
  ! --- module types
  ! ---------------------
  type, bind(c) :: ghex_message
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_message


  ! ---------------------
  ! --- module C interfaces
  ! ---------------------
  interface

     type(ghex_message) function ghex_message_new(size, allocator) bind(c)
       use iso_c_binding
       import ghex_message
       integer(c_size_t), value :: size
       integer(c_int), value :: allocator
     end function ghex_message_new

     subroutine ghex_message_zero(message) bind(c)
       use iso_c_binding
       import ghex_message
       type(ghex_message), value :: message
     end subroutine ghex_message_zero

     type(c_ptr) function ghex_message_data_wrapped(message, size) bind(c, name='ghex_message_data')
       use iso_c_binding
       import ghex_message
       type(ghex_message), value :: message
       integer(c_size_t), intent(out) :: size
     end function ghex_message_data_wrapped
  end interface


  ! ---------------------
  ! --- generic ghex interfaces
  ! ---------------------
  interface ghex_free
     subroutine ghex_message_free(message) bind(c)
       use iso_c_binding
       import ghex_message
       ! reference, not a value - fortran variable is reset to null 
       type(ghex_message) :: message
     end subroutine ghex_message_free
  end interface ghex_free

CONTAINS

  function ghex_message_data(message)
    use iso_c_binding
    type(ghex_message), value :: message
    integer(1), dimension(:), pointer :: ghex_message_data

    type(c_ptr) :: c_data
    integer(c_size_t) :: size

    ! get the data pointer
    c_data = ghex_message_data_wrapped(message, size)
    call c_f_pointer(c_data, ghex_message_data, [size])
  end function ghex_message_data

END MODULE ghex_message_mod
