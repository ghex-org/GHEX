#ifndef _THREADS_HPP
#define _THREADS_HPP

using thread_rank_type = int;

#ifdef USE_OPENMP

#define THREAD_MODE_MULTIPLE
#include <omp.h>
#define DO_PRAGMA(x) _Pragma(#x)

#define GET_THREAD_NUM()  omp_get_thread_num()
#define GET_NUM_THREADS() omp_get_num_threads()
#define IN_PARALLEL()     omp_in_parallel()

#define DECLARE_THREAD_PRIVATE(name) DO_PRAGMA(omp threadprivate(name))
#define THREAD_BARRIER()             DO_PRAGMA(omp barrier)
#define THREAD_MASTER()              DO_PRAGMA(omp master)
#define THREAD_PARALLEL_BEG() DO_PRAGMA(omp parallel)
#define THREAD_PARALLEL_END() 
#define THREAD_IS_MT 1

#else

#define GET_THREAD_NUM()  0
#define GET_NUM_THREADS() 1
#define IN_PARALLEL()     0

#define DECLARE_THREAD_PRIVATE(name)
#define THREAD_BARRIER()              
#define THREAD_MASTER()               
#define THREAD_PARALLEL_BEG()
#define THREAD_PARALLEL_END() 
#define THREAD_IS_MT 0

#endif /* USE_OPENMP */

#endif /* _THREADS_HPP */
