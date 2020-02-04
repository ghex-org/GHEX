#include "context_bind.hpp"
#include <mpi.h>

context_uptr_type context;

extern "C"
void ghex_init(int nthreads, MPI_Fint fcomm)
{
    /* the fortran-side mpi communicator must be translated to C */
    MPI_Comm ccomm = MPI_Comm_f2c(fcomm);
    
    context = ghex::tl::context_factory<transport,threading>::create(nthreads, ccomm);
}

extern "C"
void ghex_finalize()
{
    context.reset();
}

extern "C"
void* ghex_get_communicator()
{
    auto token = context->get_token();
    auto comm  = context->get_communicator(token);
    return new ghex::bindings::obj_wrapper(std::move(comm));
}
