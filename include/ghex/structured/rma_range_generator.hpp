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
#include "../rma/access_guard.hpp"
#include "./rma_range.hpp"
#include "./rma_put.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

/** @brief A range generator will generate target and source range template types based on the field
  * template paramter. The generated ranges take themselves 2 template paramters, the first of which
  * is the range factory which allows (de-) serialization and transport of range types.
  * @tparam Field the field type */
template<typename Field>
struct rma_range_generator
{
    // the range type to be used for this field
    using range_type = rma_range<Field>;

    /** @brief This class represents the target range of a halo exchange operation. It is
     * referencing a local target field (the endpoint of the put) to which it has direct memory
     * access. During construction and send() member function, it serializes the target range and
     * sends it to its remote partner.
     * It is also responsible for creating a synchronization point (access guard) which will be used
     * in RMA exchanges. The access guard, along with the rma handle of the field and the range,
     * will be serialized and sent to the remote partner.
     * @tparam RangeFactory the factory type which knows about all possible range types
     * @tparam Communicator the communicator type */
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

        void send()
        {
            m_request.wait();
        }

        void start_target_epoch()
        {
            m_local_guard.start_target_epoch();
        }

        void end_target_epoch()
        {
            m_local_guard.end_target_epoch();
        }
    };

    /** @brief This class represents the source range of a halo exchange operation. It is
     * referencing a local source field (starting point of the put) to which it has direct memory
     * access. During construction and recv() member function, it receives a target range from its
     * remote partner and deserializes it into a generic range type, which exposes synchronization
     * functions for RMA. The actual type can be recovered from generic range type by type injection
     * through the range_factory.
     * @tparam RangeFactory the factory type which knows about all possible range types
     * @tparam Communicator the communicator type */
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
        std::vector<unsigned char> m_archive;

        template<typename IterationSpace>
        source_range(const Communicator& comm, const Field& f,
            const IterationSpace& is, rank_type src, tag_type tag)
        : m_comm{comm}
        , m_local_range{f, is.local().first(), is.local().last()-is.local().first()+1}
        , m_src{src}
        , m_tag{tag}
        {
            m_archive.resize(RangeFactory::serial_size);
            m_request = m_comm.recv(m_archive, m_src, m_tag);
        }

        void recv()
        {
            m_request.wait();
            // creates a traget range
            m_remote_range = RangeFactory::deserialize(m_archive.data());
            RangeFactory::call_back_with_type(m_remote_range, [this] (auto& r)
            {
                init(r, m_remote_range);
            });
        }

        void put()
        {
            m_remote_range.start_source_epoch();
            RangeFactory::call_back_with_type(m_remote_range, [this] (auto& r)
            {
                put(r);
            });
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
// specialization of the range_traits for the above range generator
template<>
struct range_traits<structured::rma_range_generator>
{
    // determines what is local
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
