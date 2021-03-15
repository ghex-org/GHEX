MODULE ghex_request_mod
  use iso_c_binding

  ! should be established and defined by cmake, also for request_bind.hpp
#define GHEX_REQUEST_SIZE 24
  implicit none

  type, bind(c) :: ghex_request
     integer(c_int8_t) :: data(GHEX_REQUEST_SIZE) = 0
  end type ghex_request
  
  interface
     
     logical(c_bool) function ghex_request_test(request) bind(c)
       use iso_c_binding
       import ghex_request
       type(ghex_request) :: request
     end function ghex_request_test
     
     logical(c_bool) function ghex_request_cancel(request) bind(c)
       use iso_c_binding
       import ghex_request
       type(ghex_request) :: request
     end function ghex_request_cancel

  end interface

CONTAINS
  
  subroutine ghex_request_init(request)
    type(ghex_request) :: request
    request%data = 0
  end subroutine ghex_request_init
  
END MODULE ghex_request_mod
