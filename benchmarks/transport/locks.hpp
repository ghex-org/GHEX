#ifndef _LOCKS_HPP
#define _LOCKS_HPP

#define DO_PRAGMA(x) _Pragma(#x)

#ifdef THREAD_MODE_MULTIPLE

#define USE_OPENMP_LOCKS
#ifdef USE_OPENMP_LOCKS
#define CRITICAL_BEGIN(name) DO_PRAGMA(omp critical(name))
#define CRITICAL_END(name)
#else
// TODO: pthread locks
#endif

#else
#define CRITICAL_BEGIN(name)
#define CRITICAL_END(name)   
#endif

#endif /* _LOCKS_HPP */
