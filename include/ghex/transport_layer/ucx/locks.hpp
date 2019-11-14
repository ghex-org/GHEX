#ifndef _LOCKS_HPP
#define _LOCKS_HPP

#ifdef THREAD_MODE_MULTIPLE

//#define USE_OPENMP_LOCKS
//#define USE_PTHREAD_LOCKS
#define USE_PTHREAD_SPIN
//#define USE_STD_LOCKS

#if defined USE_OPENMP_LOCKS
#define DO_PRAGMA(x) _Pragma(#x)
#define CRITICAL_BEGIN(name) DO_PRAGMA(omp critical(name))
#define CRITICAL_END(name)

#elif defined USE_STD_LOCKS

#include <mutex>
using lock_t = std::mutex;

#define LOCK_INIT(l)
#define LOCK_DEL(l)

#define CRITICAL_BEGIN(name)				\
    {							\
    std::lock_guard<std::mutex> guard_##name(name);	\
    
#define CRITICAL_END(name)			\
    }						\

#elif defined USE_PTHREAD_LOCKS

#include <pthread.h>

using lock_t = pthread_mutex_t;

#define LOCK_INIT(l)					\
    do {						\
	pthread_mutex_init(&l, NULL);			\
    } while(0);

#define LOCK_DEL(l)                             \
    do {                                        \
	pthread_mutex_destroy(&l);		\
    } while(0);

#define LOCK(l)						\
    do {						\
	if(l##_owner == 0) {				\
	    pthread_mutex_lock(&l);			\
	}						\
	l##_owner += 1;					\
    } while(0);

#define UNLOCK(l)					\
    do {						\
	l##_owner -= 1;					\
	if(l##_owner == 0) {				\
	    pthread_mutex_unlock(&l);			\
	}						\
    } while(0);

#define CRITICAL_BEGIN(name) LOCK(name)
#define CRITICAL_END(name) UNLOCK(name)

#elif defined USE_PTHREAD_SPIN

#include <pthread.h>

using lock_t = pthread_spinlock_t;

#define LOCK_INIT(l)					\
    do {						\
	pthread_spin_init(&l, PTHREAD_PROCESS_PRIVATE);	\
    } while(0);

#define LOCK_DEL(l)                             \
    do {                                        \
	pthread_spin_destroy(&l);		\
    } while(0);

#define LOCK(l)						\
    do {						\
	if(l##_owner == 0) {				\
	    pthread_spin_lock(&l);			\
	}						\
	l##_owner += 1;					\
    } while(0);

#define UNLOCK(l)					\
    do {						\
	l##_owner -= 1;					\
	if(l##_owner == 0) {				\
	    pthread_spin_unlock(&l);			\
	}						\
    } while(0);

#define CRITICAL_BEGIN(name) LOCK(name)
#define CRITICAL_END(name) UNLOCK(name)

#else

/* no locking */
using lock_t = int;
#define LOCK_INIT(l)
#define CRITICAL_BEGIN(name)
#define CRITICAL_END(name)   

#endif /* USE_OPENMP_LOCKS */

#else /* THREAD_MODE_MULTIPLE */

#define CRITICAL_BEGIN(name)
#define CRITICAL_END(name)

#endif /* THREAD_MODE_MULTIPLE */

#endif /* _LOCKS_HPP */
