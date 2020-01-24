#include "request_bind.hpp"
#include "future_bind.hpp"
#include "obj_wrapper.hpp"
#include <iostream>
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


/* fortran-side user callback */
typedef void (*f_callback)(void *mesg, int rank, int tag);

struct callback {
    f_callback cb;
    callback(f_callback pcb) : cb{pcb} {}
    void operator()(communicator_type::message_type message, int rank, int tag) {
        if(cb) cb(&message, rank, tag);
    }
};

extern "C"
void comm_delete(ghex::bindings::obj_wrapper **wrapper_ref)
{
    ghex::bindings::obj_wrapper *wrapper = *wrapper_ref;

    // clear the fortran-side variable
    *wrapper_ref = nullptr;
    delete wrapper;
}

extern "C"
int comm_rank(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_safe<communicator_type>(wrapper)->rank();
}

extern "C"
int comm_size(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_safe<communicator_type>(wrapper)->size();
}

extern "C"
int comm_progress(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_safe<communicator_type>(wrapper)->progress();
}

extern "C"
void comm_barrier(ghex::bindings::obj_wrapper *wrapper)
{
    ghex::bindings::get_object_ptr_safe<communicator_type>(wrapper)->barrier();
}

extern "C"
void comm_post_send(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, frequest_type *ffut)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    auto fut = comm->send(*wmessage, rank, tag);
    new(ffut->data) decltype(fut)(std::move(fut));
}

extern "C"
void comm_post_send_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb, frequest_type *freq)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    auto req = comm->send(*wmessage, rank, tag, callback{cb});
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}

extern "C"
void comm_send_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message **wmessage_ref, int rank, int tag, f_callback cb, frequest_type *freq)
{
    ghex::tl::cb::any_message *wmessage;
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    wmessage = *wmessage_ref;
    auto req = comm->send(std::move(*wmessage), rank, tag, callback{cb});
    *wmessage_ref = nullptr;
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}

extern "C"
void comm_post_recv(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, ffuture_type *ffut)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    auto fut = comm->recv(*wmessage, rank, tag);
    new(ffut->data) decltype(fut)(std::move(fut));
}

extern "C"
void comm_post_recv_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb, frequest_type *freq)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    auto req = comm->recv(*wmessage, rank, tag, callback{cb});
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}

extern "C"
void comm_recv_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message **wmessage_ref, int rank, int tag, f_callback cb, frequest_type *freq)
{
    ghex::tl::cb::any_message *wmessage;
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    wmessage = *wmessage_ref;
    auto req = comm->recv(std::move(*wmessage), rank, tag, callback{cb});
    *wmessage_ref = nullptr;
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}

extern "C"
void comm_resubmit_recv(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb, frequest_type *freq)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    auto req = comm->recv(std::move(*wmessage), rank, tag, callback{cb});
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}
