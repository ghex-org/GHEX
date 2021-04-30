#include "obj_wrapper.hpp"
#include <iostream>
#include <cstring> 
#include <vector>
#include <ghex/transport_layer/message_buffer.hpp>
#include <ghex/transport_layer/callback_utils.hpp>
#include <ghex/allocator/unified_memory_allocator.hpp>

namespace ghex = gridtools::ghex;
using namespace gridtools::ghex::fhex;

namespace gridtools {
    namespace ghex {
        namespace fhex {

            typedef enum 
            {
                 GhexAllocatorHost=1,
                 GhexAllocatorDevice=2
            } ghex_allocator_type;

            using host_allocator_type = ghex::allocator::cuda::unified_memory_allocator<unsigned char>;
        }
    }
}

extern "C"
void *ghex_message_new(std::size_t size, int allocator_type)
{
    void *wmessage = nullptr;

    switch(allocator_type){
    case GhexAllocatorHost:
	{
	    ghex::tl::message_buffer<host_allocator_type> msg{size, host_allocator_type{}};
            wmessage = new ghex::tl::cb::any_message{std::move(msg)};
	    break;
	}
    case GhexAllocatorDevice:
	{
	    std::cerr << "BINDINGS: " << __FUNCTION__ << ": DEVICE allocator not yet implemented\n";
            std::terminate();
	    break;
	}
    default:
	{
	    std::cerr << "BINDINGS: " << __FUNCTION__ << ": unsupported allocator type: " << allocator_type << "\n";
            std::terminate();
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
