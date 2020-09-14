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
#ifndef INCLUDED_GHEX_RMA_SHMEM_HANDLE_HPP
#define INCLUDED_GHEX_RMA_SHMEM_HANDLE_HPP

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include "../locality.hpp"
#include "../uuid.hpp"

namespace gridtools {
namespace ghex {
namespace rma {
namespace shmem {

struct info
{
    bool m_on_gpu;
    uuid::info m_id;
};

struct local_data_holder
{
    bool m_on_gpu;
    uuid m_id;
    boost::interprocess::shared_memory_object m_shmem_object;
    boost::interprocess::mapped_region m_region;
    
    local_data_holder(void** ptr, unsigned int size, bool on_gpu)
    : m_on_gpu{on_gpu}
    , m_id{}
    {
        *ptr = nullptr;
        // aquire the rma resource
        if (!m_on_gpu)
        {
            m_shmem_object = boost::interprocess::shared_memory_object{
                boost::interprocess::create_only, 
                m_id.name().c_str(),
                boost::interprocess::read_write};
            // set size
            m_shmem_object.truncate(size);
            // map the whole memory
            m_region = boost::interprocess::mapped_region{
                m_shmem_object,
                boost::interprocess::read_write};
            *ptr = m_region.get_address();
        }
    }
    
    ~local_data_holder()
    {
        if (!m_on_gpu)
            boost::interprocess::shared_memory_object::remove(m_id.name().c_str());
        
    }

    local_data_holder(const local_data_holder&) = delete;
    local_data_holder(local_data_holder&&) = delete;

    info get_info() const
    {
        return { m_on_gpu, m_id.get_info() };
    }
};

struct remote_data_holder
{
    bool m_on_gpu;
    locality m_loc;
    boost::interprocess::shared_memory_object m_shmem_object;
    boost::interprocess::mapped_region m_region;
    void* m_shmem_ptr = nullptr;
        
    remote_data_holder(const info& info_, locality loc)
    : m_on_gpu{info_.m_on_gpu}
    , m_loc{loc}
    {
        // attach rma resource
        if (!m_on_gpu && m_loc == locality::process)
        {
            m_shmem_object = boost::interprocess::shared_memory_object{
                boost::interprocess::open_only, 
                info_.m_id.m_name,
                boost::interprocess::read_write};
            m_region = boost::interprocess::mapped_region{
                m_shmem_object,
                boost::interprocess::read_write};
            m_shmem_ptr = m_region.get_address();
        }
    }

    remote_data_holder(const remote_data_holder&) = delete;
    remote_data_holder(remote_data_holder&&) = delete;

    void* get_ptr() const
    {
        return m_shmem_ptr;
    }
};

} // namespace shmem
} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_SHMEM_HANDLE_HPP */
