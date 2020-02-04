MODULE ghex_future_mod
  use iso_c_binding

  ! should be established and defined by cmake, also for request_bind.hpp
#define GHEX_FUTURE_SIZE 8
  implicit none

  type, bind(c) :: ghex_future
     integer(c_int8_t) :: data(GHEX_FUTURE_SIZE) = [0]
  end type ghex_future
  
  interface
     
     subroutine ghex_future_wait(future) bind(c)
       use iso_c_binding
       import ghex_future
       type(ghex_future) :: future
     end subroutine ghex_future_wait
     
     logical(c_bool) function ghex_future_ready(future) bind(c)
       use iso_c_binding
       import ghex_future
       type(ghex_future) :: future
     end function ghex_future_ready
     
     logical(c_bool) function ghex_future_cancel(future) bind(c)
       use iso_c_binding
       import ghex_future
       type(ghex_future) :: future
     end function ghex_future_cancel

  end interface

CONTAINS

  subroutine ghex_future_init(future)
    type(ghex_future) :: future
    future%data = 0
  end subroutine ghex_future_init
  
END MODULE ghex_future_mod
