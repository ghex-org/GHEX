#include "message_bind.hpp"
#include "obj_wrapper.hpp"
#include <iostream>
#include <vector>

#include <ghex/transport_layer/shared_message_buffer.hpp>

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
int comm_progress(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_safe<communicator_type>(wrapper)->progress();
}

extern "C"
void* comm_post_send(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    return new ghex::bindings::obj_wrapper(comm->send(*wmessage, rank, tag));
}

extern "C"
void* comm_post_send_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    return new ghex::bindings::obj_wrapper(comm->send(*wmessage, rank, tag, callback{cb}));
}

extern "C"
void* comm_send_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message **wmessage_ref, int rank, int tag, f_callback cb)
{
    ghex::bindings::obj_wrapper *req;
    ghex::tl::cb::any_message *wmessage;
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    wmessage = *wmessage_ref;
    req = new ghex::bindings::obj_wrapper(comm->send(std::move(*wmessage), rank, tag, callback{cb}));    
    *wmessage_ref = nullptr;
    return req;
}

extern "C"
void* comm_post_recv(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    return new ghex::bindings::obj_wrapper(comm->recv(*wmessage, rank, tag));
}

extern "C"
void* comm_post_recv_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    return new ghex::bindings::obj_wrapper(comm->recv(*wmessage, rank, tag, callback{cb}));
}

extern "C"
void* comm_recv_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message **wmessage_ref, int rank, int tag, f_callback cb)
{
    ghex::bindings::obj_wrapper *req;
    ghex::tl::cb::any_message *wmessage;
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    wmessage = *wmessage_ref;
    req = new ghex::bindings::obj_wrapper(comm->recv(std::move(*wmessage), rank, tag, callback{cb}));    
    *wmessage_ref = nullptr;
    return req;
}

extern "C"
void* comm_resubmit_recv(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb)
{
    ghex::bindings::obj_wrapper *req;
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    return new ghex::bindings::obj_wrapper(comm->recv(std::move(*wmessage), rank, tag, callback{cb}));    
}
