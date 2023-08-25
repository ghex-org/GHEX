/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <ghex/rma/locality.hpp>
#include <ghex/rma/uuid.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

namespace ghex
{
namespace rma
{
namespace shmem
{
// Below are implementations of a handle in a multi-process setting using shmem.
// Please refer to the documentation in rma/handle.hpp for further explanations.

struct info
{
    bool       m_on_gpu;
    uuid::info m_id;
};

struct local_data_holder
{
    bool                                      m_on_gpu;
    uuid                                      m_id;
    boost::interprocess::shared_memory_object m_shmem_object;
    boost::interprocess::mapped_region        m_region;

    local_data_holder(void** ptr, unsigned int size, bool on_gpu)
    : m_on_gpu{on_gpu}
    , m_id{}
    {
        *ptr = nullptr;
        // aquire the rma resource
        if (!m_on_gpu)
        {
            m_shmem_object =
                boost::interprocess::shared_memory_object{boost::interprocess::create_only,
                    m_id.name().c_str(), boost::interprocess::read_write};
            // set size
            m_shmem_object.truncate(size);
            // map the whole memory
            m_region =
                boost::interprocess::mapped_region{m_shmem_object, boost::interprocess::read_write};
            *ptr = m_region.get_address();
        }
    }

    ~local_data_holder()
    {
        if (!m_on_gpu) boost::interprocess::shared_memory_object::remove(m_id.name().c_str());
    }

    local_data_holder(const local_data_holder&) = delete;
    local_data_holder(local_data_holder&&) = delete;

    info get_info() const { return {m_on_gpu, m_id.get_info()}; }
};

struct remote_data_holder
{
    bool                                      m_on_gpu;
    locality                                  m_loc;
    boost::interprocess::shared_memory_object m_shmem_object;
    boost::interprocess::mapped_region        m_region;
    void*                                     m_shmem_ptr = nullptr;

    remote_data_holder(const info& info_, locality loc, int)
    : m_on_gpu{info_.m_on_gpu}
    , m_loc{loc}
    {
        // attach rma resource
        if (!m_on_gpu && m_loc == locality::process)
        {
            m_shmem_object = boost::interprocess::shared_memory_object{
                boost::interprocess::open_only, info_.m_id.m_name, boost::interprocess::read_write};
            m_region =
                boost::interprocess::mapped_region{m_shmem_object, boost::interprocess::read_write};
            m_shmem_ptr = m_region.get_address();
        }
    }

    remote_data_holder(const remote_data_holder&) = delete;
    remote_data_holder(remote_data_holder&&) = delete;

    void* get_ptr() const { return m_shmem_ptr; }
};

} // namespace shmem
} // namespace rma
} // namespace ghex
