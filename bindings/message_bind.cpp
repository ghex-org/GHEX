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
void *shared_message_new(std::size_t size, int allocator_type)
{
    ghex::bindings::obj_wrapper *wmessage = nullptr;

    switch(allocator_type){
    case ALLOCATOR_STD:
	{    
	    ghex::tl::shared_message_buffer<t_std_allocator> msg{size, std_allocator};
            wmessage = new ghex::bindings::obj_wrapper(std::move(msg));
	    break;
	}
    // case ALLOCATOR_PERSISTENT_STD:
    //     {
    //         ghex::tl::shared_message_buffer<t_persistent_std_allocator> msg{size};
    //         wmessage = new ghex::bindings::obj_wrapper(std::move(msg));
    //         break;
    //     }
    default:
	{
	    std::cerr << "BINDINGS: " << __FUNCTION__ << ": unsupported allocator type: " << allocator_type << "\n";
	    break;
	}
    }
    
    return wmessage;
}

extern "C"
void *shared_message_ref(ghex::bindings::obj_wrapper *wmessage)
{
    using message_type = ghex::tl::shared_message_buffer<t_std_allocator>;

    // make a copy
    message_type msg = ghex::bindings::get_object_safe<message_type>(wmessage);
    ghex::tl::cb::any_message any_msg{msg.m_message};
    return new ghex::bindings::obj_wrapper(std::move(any_msg));
}


extern "C"
void *message_new(std::size_t size, int allocator_type)
{
    ghex::bindings::obj_wrapper *wmessage = nullptr;

    switch(allocator_type){
    case ALLOCATOR_STD:
	{    
	    ghex::tl::message_buffer<t_std_allocator> msg{size, std_allocator};
            ghex::tl::cb::any_message any_msg{std::move(msg)};
            wmessage = new ghex::bindings::obj_wrapper(std::move(any_msg));
	    break;
	}
    // case ALLOCATOR_PERSISTENT_STD:
    //     {
    //         ghex::tl::shared_message_buffer<t_persistent_std_allocator> msg{size};
    //         wmessage = new ghex::bindings::obj_wrapper(std::move(msg));
    //         break;
    //     }
    default:
	{
	    std::cerr << "BINDINGS: " << __FUNCTION__ << ": unsupported allocator type: " << allocator_type << "\n";
	    break;
	}
    }
    
    return wmessage;
}

extern "C"
void message_delete(ghex::bindings::obj_wrapper **wmessage_ref)
{
    ghex::bindings::obj_wrapper *wmessage = *wmessage_ref;
    
    /* clear the fortran-side variable */
    *wmessage_ref = nullptr;
    delete wmessage;
}

extern "C"
int message_is_host(ghex::bindings::obj_wrapper *wmessage)
{
    /** TODO */
    return 1;
}

extern "C"
int message_use_count(ghex::bindings::obj_wrapper *wmessage)
{
    // using message_type = ghex::tl::shared_message_buffer<t_std_allocator>;
    using message_type = ghex::tl::cb::any_message;
    return ghex::bindings::get_object_ptr_safe<message_type>(wmessage)->use_count();
}

extern "C"
unsigned char *message_data(ghex::bindings::obj_wrapper *wmessage, std::size_t *size)
{
    // using message_type = ghex::tl::shared_message_buffer<t_std_allocator>;
    // *size = ghex::bindings::get_object_ptr_safe<message_type>(wmessage)->capacity();
    using message_type = ghex::tl::cb::any_message;
    *size = ghex::bindings::get_object_ptr_safe<message_type>(wmessage)->size();
    return ghex::bindings::get_object_ptr_safe<message_type>(wmessage)->data();
}
