#include "context_bind.hpp"
#include <mpi.h>


extern "C"
void *context_new(int nthreads, MPI_Fint fcomm)
{
    /* the fortran-side mpi communicator must be translated to C */
    MPI_Comm ccomm = MPI_Comm_f2c(fcomm);

    /* The factory returns a unique_ptr to the context object,
       which is what we have to store in the wrapper.
     */
    context_uptr_type context_ptr = ghex::tl::context_factory<transport,threading>::create(nthreads, ccomm);
    return new ghex::bindings::obj_wrapper(std::move(context_ptr));
}

extern "C"
void context_delete(ghex::bindings::obj_wrapper **wcontext_ref)
{
    ghex::bindings::obj_wrapper *wcontext = *wcontext_ref;

    /* clear the fortran-side variable */
    *wcontext_ref = nullptr;
    delete wcontext;
}

extern "C"
int context_rank(ghex::bindings::obj_wrapper *wcontext)
{
    return ghex::bindings::get_object_ptr_safe<context_uptr_type>(wcontext)->get()->rank();
}

extern "C"
int context_size(ghex::bindings::obj_wrapper *wcontext)
{
    return ghex::bindings::get_object_ptr_safe<context_uptr_type>(wcontext)->get()->size();
}

extern "C"
void* context_get_communicator(ghex::bindings::obj_wrapper *wcontext)
{
    /* take a pointer to the unique_ptr, which stores the context */
    context_type &context = wrapper2context(wcontext);

    auto token = context.get_token();
    auto comm  = context.get_communicator(token);
    return new ghex::bindings::obj_wrapper(std::move(comm));
}
