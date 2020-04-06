#include "context_bind.hpp"
#include "future_bind.hpp"
#include "obj_wrapper.hpp"
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>

extern "C"
void ghex_future_wait(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->wait();
}

extern "C"
bool ghex_future_ready(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->ready();
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
    // communicator_type::future<void> *pfut = (communicator_type::future<void>*)ffut;
    // auto r_it = communicator_type::future<void>::test_any(pfut, pfut+n_futures);
    // return (r_it-pfut)+1;
}

extern "C"
bool ghex_future_cancel(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->cancel();
}
