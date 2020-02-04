#ifndef GHEX_FORTRAN_CONTEXT_BIND_INCLUDED_HPP
#define GHEX_FORTRAN_CONTEXT_BIND_INCLUDED_HPP

#include "obj_wrapper.hpp"
#include <vector>

namespace ghex = gridtools::ghex;

#ifdef GHEX_USE_OPENMP

/* OpenMP */
#include <ghex/threads/omp/primitives.hpp>
using threading    = ghex::threads::omp::primitives;
#else

/* no multithreading */
#include <ghex/threads/none/primitives.hpp>
using threading    = ghex::threads::none::primitives;
#endif


#ifdef GHEX_USE_UCP

/* UCX backend */
#include <ghex/transport_layer/ucx/context.hpp>
using transport    = ghex::tl::ucx_tag;
#else

/* fallback MPI backend */
#include <ghex/transport_layer/mpi/context.hpp>
using transport    = ghex::tl::mpi_tag;
#endif

using context_type      = ghex::tl::context<transport, threading>;
using context_uptr_type = std::unique_ptr<context_type>;
using communicator_type = context_type::communicator_type;

extern context_uptr_type context;

#endif /* GHEX_FORTRAN_CONTEXT_BIND_INCLUDED_HPP */
