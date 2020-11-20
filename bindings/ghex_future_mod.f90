MODULE ghex_future_mod
  use iso_c_binding

  ! should be established and defined by cmake, also for request_bind.hpp
#define GHEX_FUTURE_SIZE 8
#define GHEX_FUTURE_MULTI_SIZE 24
  implicit none

  type, bind(c) :: ghex_future
     integer(c_int8_t) :: data(GHEX_FUTURE_SIZE) = 0
  end type ghex_future
  
  type, bind(c) :: ghex_future_multi
     integer(c_int8_t) :: data(GHEX_FUTURE_MULTI_SIZE) = 0
  end type ghex_future_multi
  
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
     
     logical(c_bool) function ghex_future_multi_ready(future) bind(c)
       use iso_c_binding
       import ghex_future_multi
       type(ghex_future_multi) :: future
     end function ghex_future_multi_ready
     
     integer(c_int) function ghex_future_test_any_wrapped(futures, n_futures) bind(c, name="ghex_future_test_any")
       use iso_c_binding
       type(c_ptr), value :: futures
       integer(c_int), value :: n_futures
     end function ghex_future_test_any_wrapped
     
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

  integer(c_int) function ghex_future_test_any(futures)
    type(ghex_future), dimension(:), target :: futures
    ghex_future_test_any = ghex_future_test_any_wrapped(c_loc(futures), size(futures))
  end function ghex_future_test_any
  
END MODULE ghex_future_mod
