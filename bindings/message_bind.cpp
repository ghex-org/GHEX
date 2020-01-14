#define GHEX_DEBUG_LEVEL 2

#include "message_bind.hpp"
#include <iostream>
#include <vector>

/** Statically construct all allocator objects */
t_std_allocator std_allocator;
// t_persistent_std_allocator persistent_std_allocator;


extern "C"
void *shared_message_new(std::size_t size, int allocator_type)
{
    gridtools::ghex::bindings::obj_wrapper *wrapper = nullptr;

#if (GHEX_DEBUG_LEVEL == 2)
    std::cout << "BINDINGS: " << __FUNCTION__ << ": constructing a message with allocator " << allocator_type << "\n";
#endif
    
    switch(allocator_type){
    case ALLOCATOR_STD:
	{    
	    gridtools::ghex::tl::shared_message_buffer<t_std_allocator> msg{size};
	    wrapper = new gridtools::ghex::bindings::obj_wrapper(std::move(msg));
	    break;
	}
    // case ALLOCATOR_PERSISTENT_STD:
    //     {
    //         gridtools::ghex::tl::shared_message_buffer<t_persistent_std_allocator> msg{size};
    //         wrapper = new gridtools::ghex::bindings::obj_wrapper(std::move(msg));
    //         break;
    //     }
    default:
	{
	    std::cerr << "BINDINGS: " << __FUNCTION__ << ": unknown allocator type: " << allocator_type << "\n";
	    break;
	}
    }
    
    return wrapper;
}

extern "C"
void shared_message_delete(gridtools::ghex::bindings::obj_wrapper **wrapper_ref)
{
    gridtools::ghex::bindings::obj_wrapper *wrapper = *wrapper_ref;
    
    /* clear the fortran-side variable */
    *wrapper_ref = nullptr;
    delete wrapper;
}

extern "C"
int shared_message_is_host(gridtools::ghex::bindings::obj_wrapper *wrapper)
{
    /** TODO */
    return 1;
}

extern "C"
int shared_message_use_count(gridtools::ghex::bindings::obj_wrapper *wrapper)
{
    SHARED_MESSAGE_CALL(wrapper, {
	    return msg.use_count()-1;
	} );
    return -1;
}

extern "C"
unsigned char *shared_message_data(gridtools::ghex::bindings::obj_wrapper *wrapper, std::size_t *size)
{
    SHARED_MESSAGE_CALL(wrapper, {
	    *size = msg.capacity();
	    return msg.data();
	} );
    return nullptr;
}

extern "C"
void shared_message_resize(gridtools::ghex::bindings::obj_wrapper *wrapper, std::size_t size)
{
    SHARED_MESSAGE_CALL(wrapper, {
	    msg.resize(size);
	} );
}
