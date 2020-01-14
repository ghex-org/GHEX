#ifndef MESSAGE_BIND_HPP
#define MESSAGE_BIND_HPP

#include "obj_wrapper.hpp"
#include <transport_layer/mpi/message.hpp>
#include <allocator/persistent_allocator.hpp>

#define ALLOCATOR_STD             1
#define ALLOCATOR_PERSISTENT_STD  2
#define ALLOCATOR_HOST            3
#define ALLOCATOR_PERSISTENT_HOST 4
#define ALLOCATOR_GPU             5
#define ALLOCATOR_PERSISTENT_GPU  6

using t_std_allocator = std::allocator<unsigned char>;
using t_persistent_std_allocator = ghex::allocator::persistent_allocator<unsigned char, std::allocator<unsigned char>>;

/** a macro that casts a void* to the correct templated message type
 *  and executes code for that message
 */
#define SHARED_MESSAGE_CALL(wrapper, code)				\
    if(wrapper->type_info() == typeid(gridtools::ghex::mpi::shared_message<t_std_allocator>)) \
	{								\
	    using message_type = gridtools::ghex::mpi::shared_message<t_std_allocator>; \
	    message_type msg{gridtools::ghex::bindings::get_object_safe<message_type>(wrapper)}; \
	    code;							\
	} else if(wrapper->type_info() == typeid(gridtools::ghex::mpi::shared_message<t_persistent_std_allocator>)) \
	{								\
	    using message_type = gridtools::ghex::mpi::shared_message<t_persistent_std_allocator>; \
	    message_type msg{gridtools::ghex::bindings::get_object_safe<message_type>(wrapper)}; \
	    code;							\
	} else								\
	{								\
	    std::cerr << "BINDINGS: " << __FUNCTION__ << ": unknown message type " << wrapper->type_info().name() << "\n"; \
	}								\
	
#endif /* MESSAGE_BIND_HPP */
