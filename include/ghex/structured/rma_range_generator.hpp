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
#ifndef INCLUDED_GHEX_STRUCTURED_RMA_RANGE_GENERATOR_HPP
#define INCLUDED_GHEX_STRUCTURED_RMA_RANGE_GENERATOR_HPP

#include "../arch_list.hpp"
#include "../rma/range_traits.hpp"
#include "../rma/access_guard2.hpp"
#include "./rma_range.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

template<typename Field>
struct rma_range_generator
{
    using range_type = rma_range<Field>;

    template<typename RangeFactory, typename Communicator>
    struct target_range
    {
        using rank_type = typename Communicator::rank_type;
        using tag_type = typename Communicator::tag_type;

        Communicator m_comm;
        rma::local_access_guard m_local_guard;
        range_type m_local_range;
        rank_type m_dst;
        tag_type m_tag;
        typename Communicator::template future<void> m_request;
        std::vector<unsigned char> m_archive;

        template<typename IterationSpace>
        target_range(const Communicator& comm, const Field& f, rma::info field_info,
            const IterationSpace& is, rank_type dst, tag_type tag, rma::locality loc)
        : m_comm{comm}
        , m_local_guard{loc}
        , m_local_range{f, is.local().first(), is.local().last()-is.local().first()+1}
        , m_dst{dst}
        , m_tag{tag}
        {
            m_archive.resize(RangeFactory::serial_size);
            m_archive = RangeFactory::serialize(field_info, m_local_guard, m_local_range);
            m_request = m_comm.send(m_archive, m_dst, m_tag);
        }

        void start_target_epoch()
        {
            m_local_guard.start_target_epoch();
        }
        void end_target_epoch()
        {
            m_local_guard.end_target_epoch();
        }

        void send()
        {
            m_request.wait();
        }
    };

    template<typename RangeFactory, typename Communicator>
    struct source_range
    {
        using field_type = Field;
        using info = rma::info;
        using rank_type = typename Communicator::rank_type;
        using tag_type = typename Communicator::tag_type;

        Communicator m_comm;
        range_type m_local_range;
        typename RangeFactory::range_type m_remote_range;
        rank_type m_src;
        tag_type m_tag;
        typename Communicator::template future<void> m_request;
        std::vector<unsigned char> m_buffer;

        template<typename IterationSpace>
        source_range(const Communicator& comm, const Field& f,
            const IterationSpace& is, rank_type src, tag_type tag)
        : m_comm{comm}
        , m_local_range{f, is.local().first(), is.local().last()-is.local().first()+1}
        , m_src{src}
        , m_tag{tag}
        {
            m_buffer.resize(RangeFactory::serial_size);
            m_request = m_comm.recv(m_buffer, m_src, m_tag);
        }

        void recv()
        {
            m_request.wait();
            // creates a traget range
            m_remote_range = RangeFactory::deserialize(m_buffer.data());
            RangeFactory::call_back_with_type(m_remote_range, [this] (auto& r)
            {
                init(r, m_remote_range);
            });
        }

        void put()
        {
            m_remote_range.start_source_epoch();
            RangeFactory::put(*this, m_remote_range);
            m_remote_range.end_source_epoch();
        }

        template<typename TargetRange>
        void put(TargetRange& tr)
        {
            ::gridtools::ghex::structured::put(m_local_range, tr);
        }

    private:

        template<typename TargetRange>
        void init(TargetRange& tr, rma::range& r)
        {
            using T = typename TargetRange::value_type;
            tr.m_field.set_data((T*)r.get_ptr());
        }
    };
};

} // namespace structured

namespace rma {
template<>
struct range_traits<structured::rma_range_generator>
{
    template<typename Communicator>
    static locality is_local(Communicator comm, int remote_rank)
    {
        if (comm.rank() == remote_rank) return locality::thread;
#ifdef GHEX_USE_XPMEM
        else if (comm.is_local(remote_rank)) return locality::process;
#endif /* GHEX_USE_XPMEM */
        else return locality::remote;
    }
};
} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_RANGE_GENERATOR_HPP */
