#include "future_bind.hpp"
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

using context_type = ghex::tl::context<transport, threading>;
using communicator_type = context_type::communicator_type;

extern "C"
void future_wait(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->wait();
}

extern "C"
bool future_ready(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->ready();
}

extern "C"
bool future_cancel(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->cancel();
}
