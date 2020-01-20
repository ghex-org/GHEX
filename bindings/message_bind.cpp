#include "message_bind.hpp"
#include <iostream>
#include <vector>

#include <ghex/transport_layer/message_buffer.hpp>
#include <ghex/transport_layer/callback_utils.hpp>

namespace ghex = gridtools::ghex;

/** Statically construct all allocator objects */
t_std_allocator std_allocator;
// t_persistent_std_allocator persistent_std_allocator;


extern "C"
void *message_new(std::size_t size, int allocator_type)
{
    void *wmessage = nullptr;

    switch(allocator_type){
    case ALLOCATOR_STD:
	{    
	    ghex::tl::message_buffer<t_std_allocator> msg{size, std_allocator};
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
void message_delete(ghex::tl::cb::any_message **wmessage_ref)
{
    ghex::tl::cb::any_message *wmessage = *wmessage_ref;
    
    /* clear the fortran-side variable */
    *wmessage_ref = nullptr;
    delete wmessage;
}

extern "C"
unsigned char *message_data(ghex::tl::cb::any_message *wmessage, std::size_t *size)
{
    *size = wmessage->size();
    return wmessage->data();
}
