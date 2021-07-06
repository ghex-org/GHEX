MODULE ghex_future_mod
  use iso_c_binding

  ! should be established and defined by cmake, also for request_bind.hpp
  implicit none
#include "ghex_config.h"

  type, bind(c) :: ghex_future
     integer(c_int8_t) :: data(GHEX_FUTURE_SIZE) = 0
  end type ghex_future
  
  type, bind(c) :: ghex_future_multi
     integer(c_int8_t) :: data(GHEX_FUTURE_MULTI_SIZE) = 0
  end type ghex_future_multi
  
  interface
     
     subroutine ghex_future_wait_single(future) bind(c)
       use iso_c_binding
       import ghex_future
       type(ghex_future) :: future
     end subroutine ghex_future_wait_single
     
     subroutine ghex_future_wait_multi(future) bind(c)
       use iso_c_binding
       import ghex_future_multi
       type(ghex_future_multi) :: future
     end subroutine ghex_future_wait_multi
     
     logical(c_bool) function ghex_future_ready_single(future) bind(c)
       use iso_c_binding
       import ghex_future
       type(ghex_future) :: future
     end function ghex_future_ready_single
     
     logical(c_bool) function ghex_future_ready_multi(future) bind(c)
       use iso_c_binding
       import ghex_future_multi
       type(ghex_future_multi) :: future
     end function ghex_future_ready_multi
     
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

  interface ghex_future_init
     procedure :: ghex_future_init_single
     procedure :: ghex_future_init_multi
  end interface ghex_future_init

  interface ghex_future_wait
     procedure :: ghex_future_wait_single
     procedure :: ghex_future_wait_multi
  end interface ghex_future_wait

  interface ghex_future_ready
     procedure :: ghex_future_ready_single
     procedure :: ghex_future_ready_multi
  end interface ghex_future_ready

CONTAINS

  subroutine ghex_future_init_single(future)
    type(ghex_future) :: future
    future%data = 0
  end subroutine ghex_future_init_single

  subroutine ghex_future_init_multi(future)
    type(ghex_future_multi) :: future
    future%data = 0
  end subroutine ghex_future_init_multi

  integer(c_int) function ghex_future_test_any(futures)
    type(ghex_future), dimension(:), target :: futures
    ghex_future_test_any = ghex_future_test_any_wrapped(c_loc(futures), size(futures))
  end function ghex_future_test_any
  
END MODULE ghex_future_mod
