MODULE ghex_future_mod
  use iso_c_binding

  implicit none

  type, bind(c) :: ghex_future
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_future
  
  interface
     
     subroutine future_wait(future) bind(c)
       use iso_c_binding
       import ghex_future
       type(ghex_future), value :: future
     end subroutine future_wait
     
     logical(c_bool) function future_ready(future) bind(c)
       use iso_c_binding
       import ghex_future
       type(ghex_future), value :: future
     end function future_ready
     
     logical(c_bool) function future_cancel(future) bind(c)
       use iso_c_binding
       import ghex_future
       type(ghex_future), value :: future
     end function future_cancel

  end interface

CONTAINS
  
END MODULE ghex_future_mod
