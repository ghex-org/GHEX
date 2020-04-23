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
#ifndef INCLUDED_GHEX_UTILS_HPP
#define INCLUDED_GHEX_UTILS_HPP

#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#include <string.h>

template<typename Msg>
void make_zero(Msg& msg) {
    for (auto& c : msg)
	c = 0;
}

void bind_to_core(int thrid)
{
    cpu_set_t cpu_mask;
    pid_t tid = syscall(SYS_gettid);
    CPU_ZERO(&cpu_mask);
    CPU_SET(thrid, &cpu_mask);
    if (sched_setaffinity(tid, sizeof(cpu_mask), &cpu_mask) == -1){
        fprintf(stderr, "sched_setaffinity error : %s\n", strerror(errno));
        exit(1);
    }
}

#endif /* INCLUDED_GHEX_UTILS_HPP */

