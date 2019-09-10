#ifndef _DEBUG_H
#define _DEBUG_H

#include <string.h>
#include <time.h>

#if (GHEX_DEBUG_LEVEL == 2)

#define LOG(msg, ...)							\
    do {								\
	time_t tm = time(NULL);						\
	char *stm = ctime(&tm);						\
	stm[strlen(stm)-1] = 0;						\
	fprintf(stderr, "%s %s:%d  " msg "\n", stm, __FILE__, __LINE__, ## __VA_ARGS__); \
	fflush(stderr);							\
    } while(0);

#else
#define LOG(msg, ...)
#endif

#define ERR(msg, ...)							\
    do {								\
	time_t tm = time(NULL);						\
	char *stm = ctime(&tm);						\
	stm[strlen(stm)-1] = 0;						\
	fprintf(stderr, "%s ERROR: %s:%d  " msg "\n", stm, __FILE__, __LINE__, ## __VA_ARGS__); \
	exit(1);							\
    } while(0);

#endif /* _DEBUG_H */
