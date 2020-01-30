MODULE ghex_structured_mod
  use iso_c_binding

  use ghex_context_mod

  implicit none

  type, bind(c) :: ghex_pattern
     type(c_ptr) :: ptr = c_null_ptr
  end type ghex_pattern

  type, bind(c) :: ghex_domain_descriptor
     integer :: id
     integer :: first(3)   ! indices of the first LOCAL grid point, in global index space
     integer :: last(3)    ! indices of the last LOCAL grid point, in global index space
  end type ghex_domain_descriptor

  interface
     type(ghex_pattern) function ghex_make_pattern_wrapped(context, halo, domain_desc, ndomain_desc, periodic, &
          global_first, global_last) bind(c, name="ghex_make_pattern")
       use iso_c_binding
       import ghex_pattern, ghex_domain_descriptor, ghex_context
       type(ghex_context), value :: context
       integer(c_int), dimension(:) :: halo(6)
       type(c_ptr), value :: domain_desc
       integer(c_int), dimension(:) :: periodic(3)
       integer(c_int), value :: ndomain_desc
       integer(c_int), dimension(:) :: global_first(3), global_last(3)
     end function ghex_make_pattern_wrapped
  end interface

CONTAINS

  type(ghex_pattern) function ghex_make_pattern(context, halo, domain_desc, periodic, global_first, global_last)
    type(ghex_context) :: context
    integer, dimension(:) :: halo(6)
    type(ghex_domain_descriptor), pointer, dimension(:) :: domain_desc
    integer, dimension(:) :: periodic(3)
    integer, dimension(:) :: global_first(3), global_last(3)

    ghex_make_pattern = ghex_make_pattern_wrapped(context, halo, c_loc(domain_desc), size(domain_desc, 1), periodic, &
         global_first, global_last)
  end function ghex_make_pattern

END MODULE ghex_structured_mod
