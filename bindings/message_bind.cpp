#include "message_bind.hpp"
#include <iostream>
#include <cstring> 
#include <vector>

#include <ghex/transport_layer/message_buffer.hpp>
#include <ghex/transport_layer/callback_utils.hpp>

namespace ghex = gridtools::ghex;

// TODO how to construct the (per-thread) allocators
std_allocator_type std_allocator;

extern "C"
void *ghex_message_new(std::size_t size, int allocator_type)
{
    void *wmessage = nullptr;

    switch(allocator_type){
    case ALLOCATOR_STD:
	{
	    ghex::tl::message_buffer<std_allocator_type> msg{size, std_allocator};
            wmessage = new ghex::tl::cb::any_message{std::move(msg)};
	    break;
	}
        // case ALLOCATOR_PERSISTENT_STD:
    default:
	{
	    std::cerr << "BINDINGS: " << __FUNCTION__ << ": unsupported allocator type: " << allocator_type << "\n";
	    break;
	}
    }

    return wmessage;
}

extern "C"
void ghex_message_free(ghex::tl::cb::any_message **wmessage_ref)
{
    ghex::tl::cb::any_message *wmessage = *wmessage_ref;

    // clear the fortran-side variable
    *wmessage_ref = nullptr;
    delete wmessage;
}

extern "C"
void ghex_message_zero(ghex::tl::cb::any_message *wmessage)
{
    unsigned char* __restrict data = wmessage->data();
    std::size_t size = wmessage->size();
    std::memset(data, 0, size);
}

extern "C"
unsigned char *ghex_message_data(ghex::tl::cb::any_message *wmessage, std::size_t *size)
{
    *size = wmessage->size();
    return wmessage->data();
}
