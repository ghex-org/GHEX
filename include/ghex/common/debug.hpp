/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_GHEX_COMMON_DEBUG_HPP
#define INCLUDED_GHEX_COMMON_DEBUG_HPP

#include <string.h>
#include <time.h>

#if (GHEX_DEBUG_LEVEL == 2)

#define LOG(msg, ...)							\
    do {								\
	time_t tm = time(NULL);						\
	char *stm = ctime(&tm);						\
	stm[strlen(stm)-1] = 0;						\
	(void)fprintf(stderr, "%s %s:%d  " msg "\n", stm, __FILE__, __LINE__, ## __VA_ARGS__); \
	(void)fflush(stderr);						\
    } while(0);

#else
#define LOG(msg, ...)				\
    do { } while(0);
#endif

#define WARN(msg, ...)							\
    do {								\
	time_t tm = time(NULL);						\
	char *stm = ctime(&tm);						\
	stm[strlen(stm)-1] = 0;						\
	(void)fprintf(stderr, "%s WARNING: %s:%d  " msg "\n", stm, __FILE__, __LINE__, ## __VA_ARGS__); \
    } while(0);


#define ERR(msg, ...)							\
    do {								\
	time_t tm = time(NULL);						\
	char *stm = ctime(&tm);						\
	stm[strlen(stm)-1] = 0;						\
	(void)fprintf(stderr, "%s ERROR: %s:%d  " msg "\n", stm, __FILE__, __LINE__, ## __VA_ARGS__); \
	(void)exit(1);							\
    } while(0);

#endif /* INCLUDED_GHEX_COMMON_DEBUG_HPP */
