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

#include <ghex/common/debug.hpp>
#include <ghex/common/moved_bit.hpp>
#include "../pmi.hpp"

namespace gridtools
{
    namespace ghex
    {
        namespace tl
        {
            namespace pmi
            {
                template<>
                class pmi<pmix_tag>
                {

                private:
                    moved_bit m_moved;
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
                            throw std::runtime_error("PMIx_Init failed with code " + std::to_string(rc));
                        }
                        if(myproc.rank == 0) LOG("%d PMIx initialized", myproc.rank);

                        /* job-related info is found in our nspace, assigned to the
                         * wildcard rank as it doesn't relate to a specific rank. Setup
                         * a name to retrieve such values */
                        PMIX_PROC_CONSTRUCT(&allproc);
                        // (void)strncpy(allproc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
                        std::memcpy(allproc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
                        allproc.rank = PMIX_RANK_WILDCARD;

                        /* get the number of procs in our job */
                        if (PMIX_SUCCESS != (rc = PMIx_Get(&allproc, PMIX_JOB_SIZE, NULL, 0, &pvalue))) {
                            std::string nspace(myproc.nspace, myproc.nspace+strlen(myproc.nspace));
                            throw std::runtime_error("Client ns " + nspace + " rank " + std::to_string(myproc.rank) + 
                                ": PMIx_Get job size failed: " + std::to_string(rc) + "\n");
                        }
                        nprocs = pvalue->data.uint32;
                        PMIX_VALUE_RELEASE(pvalue);
                    }

                    pmi(const pmi&) = delete;
                    pmi(pmi&&) = default;
                    pmi& operator=(const pmi&) = delete;
                    pmi& operator=(pmi&&) = default;

                    ~pmi()
                    {
                        int rc;
                        if(m_moved) return;
                        if (PMIX_SUCCESS != (rc = PMIx_Finalize(NULL, 0))) {
                            WARN("Client ns %s rank %d:PMIx_Finalize failed: %d\n", myproc.nspace, myproc.rank, rc);
                        } else {
                            if(myproc.rank == 0) LOG("%d PMIx finalized", myproc.rank);
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

                    void set(const std::string key, const std::vector<unsigned char> data)
                    {
                        int rc;
                        pmix_value_t value;

                        PMIX_VALUE_CONSTRUCT(&value);
                        value.type = PMIX_BYTE_OBJECT;
                        value.data.bo.bytes = (char*)data.data();
                        value.data.bo.size  = data.size();
                        if (PMIX_SUCCESS != (rc = PMIx_Put(PMIX_GLOBAL, key.c_str(), &value))) {
                            std::string nspace(myproc.nspace, myproc.nspace+strlen(myproc.nspace));
                            throw std::runtime_error("Client ns " + nspace + " rank " + std::to_string(myproc.rank) + 
                                ": PMIx_Put failed: " + key + " " + std::to_string(rc) + "\n");
                        }

                        /* protect the data */
                        value.data.bo.bytes = NULL;
                        value.data.bo.size  = 0;
                        PMIX_VALUE_DESTRUCT(&value);
                        LOG("PMIx_Put on %s", key.c_str());

                        if (PMIX_SUCCESS != (rc = PMIx_Commit())) {
                            std::string nspace(myproc.nspace, myproc.nspace+strlen(myproc.nspace));
                            throw std::runtime_error("Client ns " + nspace + " rank " + std::to_string(myproc.rank) + 
                                ": PMIx_Commit failed: " + key + " " + std::to_string(rc) + "\n");
                        }
                        LOG("PMIx_Commit on %s", key.c_str());
                    }

                    std::vector<unsigned char> get(uint32_t peer_rank, const std::string key)
                    {
                        int rc;
                        pmix_proc_t proc;
                        pmix_value_t *pvalue;

                        PMIX_PROC_CONSTRUCT(&proc);
                        // (void)strncpy(proc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
                        std::memcpy(proc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
                        proc.rank = peer_rank;
                        if (PMIX_SUCCESS != (rc = PMIx_Get(&proc, key.c_str(), NULL, 0, &pvalue))) {
                            std::string nspace(myproc.nspace, myproc.nspace+strlen(myproc.nspace));
                            throw std::runtime_error("Client ns " + nspace + " rank " + std::to_string(myproc.rank) + 
                                ": PMIx_Get " + key + ": " + std::to_string(rc) + "\n");
                        }
                        if(pvalue->type != PMIX_BYTE_OBJECT){
                            std::string nspace(myproc.nspace, myproc.nspace+strlen(myproc.nspace));
                            throw std::runtime_error("Client ns " + nspace + " rank " + std::to_string(myproc.rank) + 
                                ": PMIx_Get " + key + ": " + std::to_string(rc) + "\n");
                        }

                        /* get the returned data */
                        std::vector<unsigned char> data(pvalue->data.bo.bytes, pvalue->data.bo.bytes+pvalue->data.bo.size);

                        /* free the PMIx data */
                        PMIX_VALUE_RELEASE(pvalue);
                        PMIX_PROC_DESTRUCT(&proc);

                        LOG("PMIx_get %s returned %zi bytes", key.c_str(), data.size());

                        return data;
                    }

                    void exchange()
                    {
                        int rc;
                        pmix_info_t info;
                        bool flag;

                        PMIX_INFO_CONSTRUCT(&info);
                        flag = true;
                        PMIX_INFO_LOAD(&info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);
                        if (PMIX_SUCCESS != (rc = PMIx_Fence(&allproc, 1, &info, 1))){
                            std::string nspace(myproc.nspace, myproc.nspace+strlen(myproc.nspace));
                            throw std::runtime_error("Client ns " + nspace + " rank " + std::to_string(myproc.rank) + 
                                ": PMIx_Fence failed: " + std::to_string(rc) + "\n");
                        }
                        PMIX_INFO_DESTRUCT(&info);        
                    }
                };
            } // namespace pmi
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_PMIX_HPP */
