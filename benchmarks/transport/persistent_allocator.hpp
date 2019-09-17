#ifndef INCLUDED_PERSISTENT_ALLOCATOR_HPP
#define INCLUDED_PERSISTENT_ALLOCATOR_HPP

#include <mpi.h>
#include <set>
#include <algorithm>
#include <string.h>

extern int grank;

namespace ghex {

    namespace allocator {

	/** storage for the pointer and the allocation size */
	template <typename T>
	struct persistent_pointer {
	    T* ptr;
	    std::size_t n;
	    persistent_pointer(const std::size_t size, T* p): ptr{p}, n{size}{}
	};

	/** comparator for free allocation multiset: store by (non-unique) allocation size */	
	template <typename T>
	struct compare_size {
	    bool operator() (const persistent_pointer<T> &x, const persistent_pointer<T> &y){
		return x.n < y.n;
	    }
	};

	/** comparator for used allocation set: store by (unique) pointer value */	
	template <typename T>
	struct compare_ptr {
	    bool operator() (const persistent_pointer<T> &x, const persistent_pointer<T> &y){
		return x.ptr < y.ptr;
	    }
	};

        template <typename T, typename BaseAllocator>
        struct persistent_allocator {

	    typedef T value_type;

	    BaseAllocator ba;
	    static std::multiset<persistent_pointer<T>, compare_size<T>> free_alloc;
	    static std::set<persistent_pointer<T>, compare_ptr<T>> used_alloc;

            persistent_allocator() = default;

	    void initialize(int nb, int size){
	    }

            [[nodiscard]] T* allocate(std::size_t n)
            {
		/* size threashold: this allocator is only useful for large messages */
		/* that are not sent in-flight by the backend */
		// if(n < 8192) return ba.allocate(n);

		/** look for a large enough existing allocation */
		auto existing = std::find_if(free_alloc.begin(), free_alloc.end(), [n](const persistent_pointer<T> &x){
			return n <= x.n;
		    });

		/** return one if found */
		if (existing != free_alloc.end()){
		    used_alloc.insert(*existing);
		    free_alloc.erase(existing);
#if (GHEX_DEBUG_LEVEL == 2)
		    std::cout << grank << ": allocate existing " << (std::size_t)(*existing).ptr << "\n";
#endif
		    return (*existing).ptr;
		}

		/** if no free allocations, make a new one */
		T* ptr = ba.allocate(n);
		memset(ptr, 0, sizeof(T)*n);
		used_alloc.emplace(n, ptr);
#if (GHEX_DEBUG_LEVEL == 2)
		std::cout << grank << ": allocate new " << (std::size_t)ptr << "\n";
#endif
		return ptr;
            }

            void deallocate(T* p, std::size_t n)
            {
		/* size threashold: this allocator is only useful for large messages */
		/* that are not sent in-flight by the backend */
		// if(n < 8192) return ba.deallocate(p, n);

		/** look for the allocation in used_alloc */
		auto existing = std::find_if(used_alloc.begin(), used_alloc.end(), [p](const persistent_pointer<T> &x){
			return p == x.ptr;
		    });

		/** has to exist, otherwise it is not our pointer
		 *  and the behavior is unspecified
		 */
		if(existing == used_alloc.end()){
#if (GHEX_DEBUG_LEVEL == 2)
		    std::cout << grank << ": deallocate failed for " << (std::size_t)p << ": not my pointer...\n";
#endif
		    return;
		}

		/** never really free the memory - store the allocation for future use */
		free_alloc.insert(*existing);
		used_alloc.erase(existing);
#if (GHEX_DEBUG_LEVEL == 2)
		std::cout << grank << ": deallocate, number of allocations : " << free_alloc.size() << "/" << used_alloc.size() << "\n";
#endif
            }
        };

	template <typename T, typename BA>
	std::multiset<persistent_pointer<T>, compare_size<T>>  persistent_allocator<T, BA>::free_alloc;

	template <typename T, typename BA>
	std::set<persistent_pointer<T>, compare_ptr<T>> persistent_allocator<T, BA>::used_alloc;

    } // namespace allocator
} // namespace ghex

#endif /* INCLUDED_PERSISTENT_ALLOCATOR_HPP */
