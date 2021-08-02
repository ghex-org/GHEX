#include "context_bind.hpp"
#include "future_bind.hpp"
#include "obj_wrapper.hpp"
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>

using namespace gridtools::ghex::fhex;

extern "C"
void ghex_future_wait_single(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->wait();
}

extern "C"
void ghex_future_wait_multi(ffuture_multi_type *ffut)
{
    using ftype = communicator_type::future<void>;
    std::vector<ftype> &futures = reinterpret_cast<std::vector<ftype>&>(ffut->data);
    for(long unsigned int i=0; i<futures.size(); i++){
        futures[i].wait();
    }
}

extern "C"
bool ghex_future_ready_single(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->ready();
}

extern "C"
bool ghex_future_ready_multi(ffuture_multi_type *ffut)
{
    using ftype = communicator_type::future<void>;
    bool ret = true;
    std::vector<ftype> &futures = reinterpret_cast<std::vector<ftype>&>(ffut->data);
    for(long unsigned int i=0; i<futures.size(); i++){
	if(!futures[i].ready()){
	    ret = false;
	    break;
	}
    }
    return ret;
}
 
extern "C"
int ghex_future_test_any(ffuture_type *ffut, int n_futures)
{
    static thread_local int i = 0;
    for(; i<n_futures; i++){
        communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut[i].data);
        if(future->ready()){
            return i+1;
        }
    }
    i = 0;
    return n_futures+1;
}

extern "C"
bool ghex_future_cancel(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->cancel();
}
