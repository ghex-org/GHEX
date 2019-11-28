/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef INCLUDED_GHEX_TL_PMIX_HPP
#define INCLUDED_GHEX_TL_PMIX_HPP

#include <stdio.h>
#include <pmix.h>
#include <pthread.h>

#include <string>
#include <ghex/common/debug.hpp>
#include "../pmi.hpp"

namespace gridtools
{
    namespace ghex
    {
	namespace tl
	{
	    typedef struct {
		pthread_mutex_t mutex;
		pthread_cond_t  cond;
		pmix_status_t   status;
		volatile bool   active;
	    } cblock_t;

	    template<>
	    class pmi<pmix_tag>
	    {

	    private:
		pmix_proc_t allproc;
		pmix_proc_t myproc;
		int32_t nprocs;

	    public:
		using rank_type = int;
		using size_type = int;

	    public:

		pmi()
		{
		    int rc;
		    pmix_value_t *pvalue;
			
		    if (PMIX_SUCCESS != (rc = PMIx_Init(&myproc, NULL, 0))) {
			ERR("PMIx_Init failed");
			exit(1);
		    }
		    if(myproc.rank == 0) LOG("PMIx initialized");

		    /* job-related info is found in our nspace, assigned to the
		     * wildcard rank as it doesn't relate to a specific rank. Setup
		     * a name to retrieve such values */
		    PMIX_PROC_CONSTRUCT(&allproc);
		    // (void)strncpy(allproc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
		    (void)memcpy(allproc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
		    allproc.rank = PMIX_RANK_WILDCARD;

		    /* get the number of procs in our job */
		    if (PMIX_SUCCESS != (rc = PMIx_Get(&allproc, PMIX_JOB_SIZE, NULL, 0, &pvalue))) {
			ERR("Client ns %s rank %d: PMIx_Get job size failed: %d\n", myproc.nspace, myproc.rank, rc);
			exit(1);
		    }
		    nprocs = pvalue->data.uint32;
		    PMIX_VALUE_RELEASE(pvalue);
		}

		~pmi()
		{
		    int rc;
		    if (PMIX_SUCCESS != (rc = PMIx_Finalize(NULL, 0))) {
			ERR("Client ns %s rank %d:PMIx_Finalize failed: %d\n", myproc.nspace, myproc.rank, rc);
		    } else {
			if(myproc.rank == 0) LOG("PMIx finalized");
		    }
		}
		
		rank_type rank() 
		{
		    return myproc.rank;
		}

		size_type size()
		{
		    return nprocs;
		}

		void set(const std::string key, std::vector<char> data)
		{
		    int rc;
		    pmix_value_t value;

		    PMIX_VALUE_CONSTRUCT(&value);
		    value.type = PMIX_BYTE_OBJECT;
		    value.data.bo.bytes = data.data();
		    value.data.bo.size  = data.size();
		    if (PMIX_SUCCESS != (rc = PMIx_Put(PMIX_GLOBAL, key.c_str(), &value))) {
			ERR("Client ns %s rank %d: PMIx_Put failed: %d\n", myproc.nspace, myproc.rank, rc);
		    }

		    /* protect the data */
		    value.data.bo.bytes = NULL;
		    value.data.bo.size  = 0;
		    PMIX_VALUE_DESTRUCT(&value);
		    LOG("PMIx_Put on %s", key.c_str());

		    if (PMIX_SUCCESS != (rc = PMIx_Commit())) {
			ERR("Client ns %s rank %d: PMIx_Commit failed: %d\n", myproc.nspace, myproc.rank, rc);
		    }
		    LOG("PMIx_Commit on %s", key.c_str());
		}

		std::vector<char> get_bytes(uint32_t peer_rank, const std::string key)
		{
		    int rc;
		    pmix_proc_t proc;
		    pmix_value_t *pvalue;

		    PMIX_PROC_CONSTRUCT(&proc);
		    // (void)strncpy(proc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
		    (void)memcpy(proc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
		    proc.rank = peer_rank;
		    if (PMIX_SUCCESS != (rc = PMIx_Get(&proc, key.c_str(), NULL, 0, &pvalue))) {
			ERR("Client ns %s rank %d: PMIx_Get %s: %d\n", myproc.nspace, myproc.rank, key.c_str(), rc);
		    }
		    if(pvalue->type != PMIX_BYTE_OBJECT){
			ERR("Client ns %s rank %d: PMIx_Get %s: got wrong data type\n", myproc.nspace, myproc.rank, key.c_str());
		    }

		    /* get the returned data */
		    std::vector<char> data(pvalue->data.bo.bytes, pvalue->data.bo.bytes+pvalue->data.bo.size);

		    /* free the PMIx data */
		    PMIX_VALUE_RELEASE(pvalue);
		    PMIX_PROC_DESTRUCT(&proc);

		    LOG("PMIx_get %s returned %zi bytes", key.c_str(), data.size());
		    
		    return  data;
		}
	    };
	} // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_PMIX_HPP */
