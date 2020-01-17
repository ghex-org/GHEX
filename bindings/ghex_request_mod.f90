MODULE ghex_request_mod
  use iso_c_binding

  implicit none

  type, bind(c) :: ghex_request
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_request
  
  interface
     
     logical(c_bool) function request_test(request) bind(c)
       use iso_c_binding
       import ghex_request
       type(ghex_request), value :: request
     end function request_test
     
     logical(c_bool) function request_cancel(request) bind(c)
       use iso_c_binding
       import ghex_request
       type(ghex_request), value :: request
     end function request_cancel

  end interface

CONTAINS
  
END MODULE ghex_request_mod
