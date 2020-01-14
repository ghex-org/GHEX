#include "obj_wrapper.hpp"
#include <iostream>
#include <vector>
#include <mpi.h>

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

using context_type = ghex::tl::context<transport, threading>;
using communicator_type = context_type::communicator_type;
using context_uptr_type = std::unique_ptr<context_type>;


extern "C"
void *context_new(int nthreads, MPI_Fint fcomm)
{
    ghex::bindings::obj_wrapper *wrapper = nullptr;

    /* the fortran-side mpi communicator must be translated to C */
    MPI_Comm ccomm = MPI_Comm_f2c(fcomm);

    /* The factory returns a unique_ptr to the context object,
       which is what we have to store in the wrapper.
     */
    context_uptr_type context_ptr = ghex::tl::context_factory<transport,threading>::create(nthreads, ccomm);
    wrapper = new ghex::bindings::obj_wrapper(std::move(context_ptr));
    return wrapper;
}


extern "C"
void context_delete(ghex::bindings::obj_wrapper **wrapper_ref)
{
    ghex::bindings::obj_wrapper *wrapper = *wrapper_ref;

    /* clear the fortran-side variable */
    *wrapper_ref = nullptr;
    delete wrapper;
}


extern "C"
void* context_get_communicator(ghex::bindings::obj_wrapper *wcontext)
{
    ghex::bindings::obj_wrapper *wrapper = nullptr;

    /* take a pointer to the unique_ptr, which stores the context */
    context_uptr_type *context_ptr = ghex::bindings::get_object_ptr_safe<context_uptr_type>(wcontext);

    auto token = context_ptr->get()->get_token();
    auto comm  = context_ptr->get()->get_communicator(token);
    wrapper = new ghex::bindings::obj_wrapper(std::move(comm));
    return wrapper;
}
