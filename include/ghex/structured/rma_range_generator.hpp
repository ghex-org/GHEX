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

#include "../rma/range_traits.hpp"
#include "../arch_list.hpp"
#include "./rma_range.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

template<typename Arch>
struct select_arch
{
    static auto get() noexcept { return tl::ri::host; }
};
template<>
struct select_arch<gpu>
{
    static auto get() noexcept { return tl::ri::device; }
};

template<typename Field>
struct rma_range_generator
{
    using range_type = rma_range<Field>;
    using guard_type = typename range_type::guard_type;
    using guard_view_type = typename range_type::guard_view_type;

    template<typename RangeFactory, typename Communicator>
    struct target_range
    {
        using rank_type = typename Communicator::rank_type;
        using tag_type = typename Communicator::tag_type;

        Communicator m_comm;
        guard_type m_guard;
        guard_view_type m_guard_view;
        field_view<Field> m_view;
        rank_type m_dst;
        tag_type m_tag;
        typename Communicator::template future<void> m_request;
        std::vector<unsigned char> m_archive;

        template<typename IterationSpace>
        target_range(const Communicator& comm, const Field& f, const IterationSpace& is, rank_type dst, tag_type tag, tl::ri::locality loc)
        : m_comm{comm}
        , m_guard{}
        , m_guard_view{m_guard, loc}
        , m_view{f, is.local().first(), is.local().last()-is.local().first()+1}
        , m_dst{dst}
        , m_tag{tag}
        {
            m_archive.resize(RangeFactory::serial_size);
            m_view.m_field.init_rma_local();
            m_view.m_rma_data = m_view.m_field.get_rma_data();
            m_archive = RangeFactory::serialize(rma_range<Field>{m_view,m_guard,loc});
            m_request = m_comm.send(m_archive, m_dst, m_tag);
        }

        void start_target_epoch()
        {
            m_guard_view.start_local_epoch();
        }
        void end_target_epoch()
        {
            m_guard_view.end_local_epoch();
        }

        void send()
        {
            m_request.wait();
        }

        void release()
        {
            m_view.m_field.release_rma_local();
        }
    };

    template<typename RangeFactory, typename Communicator>
    struct source_range
    {
        using field_type = Field;
        using rank_type = typename Communicator::rank_type;
        using tag_type = typename Communicator::tag_type;

        Communicator m_comm;
        field_view<Field> m_view;
        typename RangeFactory::range_type m_remote_range;
        rank_type m_src;
        tag_type m_tag;
        typename Communicator::template future<void> m_request;
        std::vector<unsigned char> m_buffer;

        template<typename IterationSpace>
        source_range(const Communicator& comm, const Field& f, const IterationSpace& is, rank_type src, tag_type tag)
        : m_comm{comm}
        , m_view{f, is.local().first(), is.local().last()-is.local().first()+1}
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
            //m_buffer.resize(m_remote_range.buffer_size());
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
            ::gridtools::ghex::structured::put(m_view, tr.m_view);
        }

        void release()
        {
        }
    };
};

} // namespace structured

namespace rma {
template<>
struct range_traits<structured::rma_range_generator>
{
    template<typename Communicator>
    static tl::ri::locality is_local(Communicator comm, int remote_rank)
    {
        if (comm.rank() == remote_rank) return tl::ri::locality::thread;
#ifdef GHEX_USE_XPMEM
        else if (comm.is_local(remote_rank)) return tl::ri::locality::process;
#endif /* GHEX_USE_XPMEM */
        else return tl::ri::locality::remote;
    }
};
} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_RANGE_GENERATOR_HPP */
