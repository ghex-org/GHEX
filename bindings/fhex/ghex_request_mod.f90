MODULE ghex_request_mod
  use iso_c_binding

  ! should be established and defined by cmake, also for request_bind.hpp
  implicit none
#include "ghex_config.h"

  type, bind(c) :: ghex_request
     integer(c_int8_t) :: data(GHEX_REQUEST_SIZE) = 0
  end type ghex_request
  
  type, bind(c) :: ghex_request_multi
     integer(c_int8_t) :: data(GHEX_REQUEST_MULTI_SIZE) = 0
  end type ghex_request_multi
  
  interface
     
     logical(c_bool) function ghex_request_test_single(request) bind(c)
       use iso_c_binding
       import ghex_request
       type(ghex_request) :: request
     end function ghex_request_test_single
     
     logical(c_bool) function ghex_request_test_multi(request) bind(c)
       use iso_c_binding
       import ghex_request_multi
       type(ghex_request_multi) :: request
     end function ghex_request_test_multi

     ! cannot cancel multi requests, as send requests in general cannot be canceled
     logical(c_bool) function ghex_request_cancel(request) bind(c)
       use iso_c_binding
       import ghex_request
       type(ghex_request) :: request
     end function ghex_request_cancel

  end interface

  interface ghex_request_init
     procedure :: ghex_request_init_single
     procedure :: ghex_request_init_multi
  end interface ghex_request_init

  interface ghex_request_test
     procedure :: ghex_request_test_single
     procedure :: ghex_request_test_multi
  end interface ghex_request_test

CONTAINS
  
  subroutine ghex_request_init_single(request)
    type(ghex_request) :: request
    request%data = 0
  end subroutine ghex_request_init_single
  
  subroutine ghex_request_init_multi(request)
    type(ghex_request_multi) :: request
    request%data = 0
  end subroutine ghex_request_init_multi
  
END MODULE ghex_request_mod
