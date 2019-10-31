#ifndef INCLUDED_GHEX_TL_UCX_UCP_LOCKS_HPP
#define INCLUDED_GHEX_TL_UCX_UCP_LOCKS_HPP

#include "locks.hpp"
#include "threads.hpp"

namespace gridtools
{
    namespace ghex
    {
	namespace tl
	{
	    namespace ucx
	    {
#ifdef THREAD_MODE_MULTIPLE
#ifndef USE_OPENMP_LOCKS

		/* shared lock */
		lock_t ucp_lock;

		/* we can be recursive in our recv call, 
		   so no need to lock if we already have it 
		*/
		int ucp_lock_owner;
		DECLARE_THREAD_PRIVATE(ucp_lock_owner)
		
#define ucp_lock ucx::ucp_lock
#else
#define ucp_lock ucp_lock
#endif
#endif /* THREAD_MODE_MULTIPLE */
	    }
	}
    }
}

#endif /* INCLUDED_GHEX_TL_UCX_UCP_LOCKS_HPP */
