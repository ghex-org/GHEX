/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_GHEX_TL_UCX_REQUEST_HPP
#define INCLUDED_GHEX_TL_UCX_REQUEST_HPP

#include "./worker.hpp"
#include <iostream>

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace ucx {

                struct request
                {
                    void*     m_ptr = nullptr;
                    worker_t* m_worker =  nullptr;
                    worker_t* m_other_worker =  nullptr;

                    request() noexcept = default;
                    
                    request(void* ptr, worker_t* worker, worker_t* other_worker) noexcept
                    : m_ptr(ptr)
                    , m_worker(worker)
                    , m_other_worker(other_worker)
                    {}

                    request(request&& other) noexcept
                    : m_ptr(other.m_ptr)
                    , m_worker(other.m_worker)
                    , m_other_worker(other.m_other_worker)
                    {
                        other.m_ptr = nullptr;
                    }

                    request& operator=(request&& other) noexcept
                    {
                        if (m_ptr)
                        {
                            //std::cout << "destructor of request" << std::endl;
                            if (m_worker->m_shared)
                            {
                                //const typename worker_t::lock_type lock(*(m_worker->m_mutex));
                                m_worker->lock();
                                ucp_request_free(m_ptr);
                                m_worker->unlock();
                            }
                            else
                                ucp_request_free(m_ptr);
                        }
                        m_ptr = other.m_ptr;
                        m_worker = other.m_worker;
                        m_other_worker = other.m_other_worker;
                        other.m_ptr = nullptr;
                        return *this;
                    }

                    request(const request&) = delete;
                    request& operator=(const request&) = delete;
                    
                    ~request() noexcept
                    {
                        if (m_ptr)
                        {
                            //std::cout << "destructor of request" << std::endl;
                            if (m_worker->m_shared)
                            {
                                m_worker->lock();
                                //const typename worker_t::lock_type lock(*(m_worker->m_mutex));
                                ucp_request_free(m_ptr);
                                m_worker->unlock();
                            }
                            else
                                ucp_request_free(m_ptr);
                        }
                    }

                    bool test_only()
                    {
                        if (!m_ptr) return true;
                        
                        if (!m_worker->m_shared)
                        {
                            // thread unsafe is ok here
                                
                            // check for completion
                            if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                            {
                                ucp_request_free(m_ptr);
                                m_ptr = nullptr;
                                return true;
                            }
                            else
                                return false;
                        }
                        else
                        {
                            m_worker->lock();
                            if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                            {
                                ucp_request_free(m_ptr);
                                m_ptr = nullptr;
                                m_worker->unlock();
                                return true;
                            }
                            else
                            {
                                m_worker->unlock();
                                return false;
                            }
                        }
                    }

                    bool ready()
                    {
                        if (!m_ptr) return true;

                        if (m_worker->m_index > 0)
                        {
                            // send request

                            if (!m_worker->m_shared)
                            {
                                // thread unsafe is ok here
                                
                                // check for completion
                                if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                                {
                                    ucp_request_free(m_ptr);
                                    m_ptr = nullptr;
                                    return true;
                                }

                                // progress a few times
                                ucp_worker_progress(m_worker->get());
                                //ucp_worker_progress(m_worker->get());
                                //ucp_worker_progress(m_worker->get());
                                
                                if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                                {
                                    ucp_request_free(m_ptr);
                                    m_ptr = nullptr;
                                    return true;
                                }

                                // progress recv worker
                                if (m_other_worker->try_lock())
                                {
                                    ucp_worker_progress(m_other_worker->get());
                                    m_other_worker->unlock();
                                }
                                return false;
                            }
                            else
                            {
                                // thread safe here
                                
                                //if (m_worker->try_lock())
                                {
                                    m_worker->lock();
                                    if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                                    {
                                        ucp_request_free(m_ptr);
                                        m_ptr = nullptr;
                                        m_worker->unlock();
                                        return true;
                                    }
                                    // progress a few times
                                    ucp_worker_progress(m_worker->get());
                                    //ucp_worker_progress(m_worker->get());
                                    //ucp_worker_progress(m_worker->get());
                                    
                                    if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                                    {
                                        ucp_request_free(m_ptr);
                                        m_ptr = nullptr;
                                        m_worker->unlock();
                                        return true;
                                    }
                                    m_worker->unlock();
                                }

                                // progress recv worker
                                if (m_other_worker->try_lock())
                                {
                                    ucp_worker_progress(m_other_worker->get());
                                    m_other_worker->unlock();
                                }
                                return false;
                            }
                        }
                        else
                        {
                            // receive request
                            // needs tread safety always
                            //if (m_worker->try_lock())
                            {
                                m_worker->lock();
                                if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                                {
                                    m_worker->unlock();
                                    return true;
                                }
                                // progress 
                                ucp_worker_progress(m_worker->get());
                                
                                if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                                {
                                    m_worker->unlock();
                                    return true;
                                }
                                m_worker->unlock();
                            }

                            // progress send worker
                            if (m_other_worker->m_shared)
                            {
                                if (m_other_worker->try_lock())
                                {
                                    ucp_worker_progress(m_other_worker->get());
                                    m_other_worker->unlock();
                                }
                            }
                            else
                            {
                                ucp_worker_progress(m_other_worker->get());
                            }
                            return false;
                        }
                    }

                    void wait()
                    {
                        if (!m_ptr) return;

                        if (m_worker->m_index > 0)
                        {
                            // send request

                            if (!m_worker->m_shared)
                            {
                                // thread unsafe is ok here
                                
                                // check for completion
                                if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                                {
                                    ucp_request_free(m_ptr);
                                    m_ptr = nullptr;
                                    return;
                                }

                                while (true)
                                {
                                    // progress a few times
                                    ucp_worker_progress(m_worker->get());
                                    //ucp_worker_progress(m_worker->get());
                                    //ucp_worker_progress(m_worker->get());
                                    
                                    if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                                    {
                                        ucp_request_free(m_ptr);
                                        m_ptr = nullptr;
                                        return;
                                    }

                                    // progress recv worker
                                    if (m_other_worker->try_lock())
                                    {
                                        ucp_worker_progress(m_other_worker->get());
                                        m_other_worker->unlock();
                                    }
                                }
                            }
                            else
                            {
                                // thread safe here
                                
                                while (true)
                                {
                                    if (m_worker->try_lock())
                                    {
                                    
                                        if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                                        {
                                            ucp_request_free(m_ptr);
                                            m_ptr = nullptr;
                                            m_worker->unlock();
                                            return;
                                        }
                                        // progress a few times
                                        ucp_worker_progress(m_worker->get());
                                        //ucp_worker_progress(m_worker->get());
                                        //ucp_worker_progress(m_worker->get());
                                        
                                        m_worker->unlock();

                                        // progress recv worker
                                        if (m_other_worker->try_lock())
                                        {
                                            ucp_worker_progress(m_other_worker->get());
                                            m_other_worker->unlock();
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            // receive request
                            // needs tread safety always

                            while (true)
                            {
                                if (m_worker->try_lock())
                                {
                                    if (ucp_request_check_status(m_ptr) != UCS_INPROGRESS)
                                    {
                                        m_worker->unlock();
                                        return;
                                    }

                                    // progress 
                                    ucp_worker_progress(m_worker->get());
                                
                                    m_worker->unlock();

                                    // progress send worker
                                    if (m_other_worker->m_shared)
                                    {
                                        if (m_other_worker->try_lock())
                                        {
                                            ucp_worker_progress(m_other_worker->get());
                                            m_other_worker->unlock();
                                        }
                                    }
                                    else
                                    {
                                        ucp_worker_progress(m_other_worker->get());
                                    }
                                }
                            }
                        }
                    }

                };

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_REQUEST_HPP */

