#include "context_bind.hpp"
#include "request_bind.hpp"
#include "future_bind.hpp"
#include "obj_wrapper.hpp"
#include "ghex_defs.hpp"
#include <iostream>
#include <vector>

/* fortran-side user callback */
typedef void (*f_callback)(void *mesg, int rank, int tag, void *user_data);

struct callback {
    f_callback cb;
    void *user_data = NULL;
    callback(f_callback pcb, void *puser_data = NULL) : cb{pcb}, user_data{puser_data} {}
    void operator() (communicator_type::message_type message, int rank, int tag) const {
        if(cb) cb(&message, rank, tag, user_data);
    }
};

struct progress_status_type {
    int num_sends = 0;
    int num_recvs = 0;
    int num_cancels = 0;

    progress_status_type(const ghex::tl::cb::progress_status &ps) :
        num_sends{ps.m_num_sends},
        num_recvs{ps.m_num_recvs},
        num_cancels{ps.m_num_cancels} 
    {}
};

extern "C"
void* ghex_comm_new()
{
    return new ghex::bindings::obj_wrapper(context->get_communicator());
}

extern "C"
int ghex_comm_rank(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_unsafe<communicator_type>(wrapper)->rank();
}

extern "C"
int ghex_comm_size(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_unsafe<communicator_type>(wrapper)->size();
}

extern "C"
progress_status_type ghex_comm_progress(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_unsafe<communicator_type>(wrapper)->progress();
}

extern "C"
void ghex_comm_barrier(ghex::bindings::obj_wrapper *wrapper, int type)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wrapper);
    switch(type){
    case(BarrierThread):
        barrier->in_node(*comm);
        break;
    case(BarrierRank):
        barrier->rank_barrier(*comm);
        break;
    default:
        (*barrier)(*comm);
        break;
    }
}

/*
   SEND requests
 */

extern "C"
void ghex_comm_post_send(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, frequest_type *ffut)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);
    if(NULL==wmessage){
	std::cerr << "ERROR: trying to submit a NULL message in " << __FUNCTION__ << ". Terminating.\n";
	std::terminate();
    }
    auto fut = comm->send(*wmessage, rank, tag);
    new(ffut->data) decltype(fut)(std::move(fut));
}

extern "C"
void ghex_comm_post_send_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb, frequest_type *freq, void *user_data)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);
    if(NULL==wmessage){
	std::cerr << "ERROR: trying to submit a NULL message in " << __FUNCTION__ << ". Terminating.\n";
	std::terminate();
    }
    auto req = comm->send(*wmessage, rank, tag, callback{cb, user_data});
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}

extern "C"
void ghex_comm_send_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message **wmessage_ref, int rank, int tag, f_callback cb, frequest_type *freq, void *user_data)
{
    ghex::tl::cb::any_message *wmessage;
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);
    wmessage = *wmessage_ref;
    if(NULL==wmessage){
	std::cerr << "ERROR: trying to submit a NULL message in " << __FUNCTION__ << ". Terminating.\n";
	std::terminate();
    }
    auto req = comm->send(std::move(*wmessage), rank, tag, callback{cb, user_data});
    *wmessage_ref = nullptr;
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}


/*
   SEND_MULTI requests
 */

extern "C"
void ghex_comm_post_send_multi(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int *ranks, int nranks, int tag, frequest_type *ffut)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);
    if(NULL==wmessage){
	std::cerr << "ERROR: trying to submit a NULL message in " << __FUNCTION__ << ". Terminating.\n";
	std::terminate();
    }
    std::vector<int> ranks_array(nranks);
    ranks_array.assign(ranks, ranks+nranks);
    auto fut = comm->send_multi(*wmessage, ranks_array, tag);
    new(ffut->data) decltype(fut)(std::move(fut));
}

extern "C"
void ghex_comm_post_send_multi_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int *ranks, int nranks, int tag, f_callback cb, frequest_type *freq, void *user_data)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);
    if(NULL==wmessage){
	std::cerr << "ERROR: trying to submit a NULL message in " << __FUNCTION__ << ". Terminating.\n";
	std::terminate();
    }
    std::vector<int> ranks_array(nranks);
    ranks_array.assign(ranks, ranks+nranks);
    auto req = comm->send_multi(*wmessage, ranks_array, tag, callback{cb, user_data});
    new(freq->data) decltype(req)(std::move(req));
}

extern "C"
void ghex_comm_send_multi_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message **wmessage_ref, int *ranks, int nranks, int tag, f_callback cb, frequest_type *freq, void *user_data)
{
    ghex::tl::cb::any_message *wmessage;
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);
    std::vector<int> ranks_array(nranks);
    ranks_array.assign(ranks, ranks+nranks);
    wmessage = *wmessage_ref;
    if(NULL==wmessage){
	std::cerr << "ERROR: trying to submit a NULL message in " << __FUNCTION__ << ". Terminating.\n";
	std::terminate();
    }
    auto req = comm->send_multi(std::move(*wmessage), ranks_array, tag, callback{cb, user_data});
    *wmessage_ref = nullptr;
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}


/*
   RECV requests
 */

extern "C"
void ghex_comm_post_recv(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, ffuture_type *ffut)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);
    if(NULL==wmessage){
	std::cerr << "ERROR: trying to submit a NULL message in " << __FUNCTION__ << ". Terminating.\n";
	std::terminate();
    }
    auto fut = comm->recv(*wmessage, rank, tag);
    new(ffut->data) decltype(fut)(std::move(fut));
}

extern "C"
void ghex_comm_post_recv_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb, frequest_type *freq, void *user_data)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);
    if(NULL==wmessage){
	std::cerr << "ERROR: trying to submit a NULL message in " << __FUNCTION__ << ". Terminating.\n";
	std::terminate();
    }
    auto req = comm->recv(*wmessage, rank, tag, callback{cb, user_data});
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}

extern "C"
void ghex_comm_recv_cb(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message **wmessage_ref, int rank, int tag, f_callback cb, frequest_type *freq, void *user_data)
{
    ghex::tl::cb::any_message *wmessage;
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);
    wmessage = *wmessage_ref;
    if(NULL==wmessage){
	std::cerr << "ERROR: trying to submit a NULL message in " << __FUNCTION__ << ". Terminating.\n";
	std::terminate();
    }
    auto req = comm->recv(std::move(*wmessage), rank, tag, callback{cb, user_data});
    *wmessage_ref = nullptr;
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}


/*
   resubmission of recv requests from inside callbacks
 */

extern "C"
void ghex_comm_resubmit_recv(ghex::bindings::obj_wrapper *wcomm, ghex::tl::cb::any_message *wmessage, int rank, int tag, f_callback cb, frequest_type *freq, void *user_data)
{
    communicator_type *comm = ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);
    if(NULL==wmessage){
	std::cerr << "ERROR: trying to submit a NULL message in " << __FUNCTION__ << ". Terminating.\n";
	std::terminate();
    }
    auto req = comm->recv(std::move(*wmessage), rank, tag, callback{cb, user_data});
    if(!freq) return;
    new(freq->data) decltype(req)(std::move(req));
}
