#include "context_bind.hpp"
#include "request_bind.hpp"
#include "future_bind.hpp"
#include "obj_wrapper.hpp"
#include <iostream>
#include <vector>


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
void* ghex_comm_new()
{
    auto token = context->get_token();
    auto comm  = context->get_communicator(token);
    return new ghex::bindings::obj_wrapper(std::move(comm));
}

extern "C"
void ghex_comm_delete(ghex::bindings::obj_wrapper **wrapper_ref)
{
    ghex::bindings::obj_wrapper *wrapper = *wrapper_ref;

    // clear the fortran-side variable
    *wrapper_ref = nullptr;
    delete wrapper;
}

extern "C"
int ghex_comm_rank(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_safe<communicator_type>(wrapper)->rank();
}

extern "C"
int ghex_comm_size(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_safe<communicator_type>(wrapper)->size();
}

extern "C"
int ghex_comm_progress(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_safe<communicator_type>(wrapper)->progress();
}

extern "C"
void ghex_comm_barrier(ghex::bindings::obj_wrapper *wrapper)
{
    ghex::bindings::get_object_ptr_safe<communicator_type>(wrapper)->barrier();
}

extern "C"
void ghex_comm_post_send(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, frequest_type *ffut)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    auto fut = comm->send(*wmessage, rank, tag);
    new(ffut->data) decltype(fut)(std::move(fut));
}

extern "C"
void ghex_comm_post_send_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb, frequest_type *freq)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    auto req = comm->send(*wmessage, rank, tag, callback{cb});
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}

extern "C"
void ghex_comm_send_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message **wmessage_ref, int rank, int tag, f_callback cb, frequest_type *freq)
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
void ghex_comm_post_recv(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, ffuture_type *ffut)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    auto fut = comm->recv(*wmessage, rank, tag);
    new(ffut->data) decltype(fut)(std::move(fut));
}

extern "C"
void ghex_comm_post_recv_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb, frequest_type *freq)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    auto req = comm->recv(*wmessage, rank, tag, callback{cb});
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}

extern "C"
void ghex_comm_recv_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message **wmessage_ref, int rank, int tag, f_callback cb, frequest_type *freq)
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
void ghex_comm_resubmit_recv(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb, frequest_type *freq)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    auto req = comm->recv(std::move(*wmessage), rank, tag, callback{cb});
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}
