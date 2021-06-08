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
#ifndef INCLUDED_GHEX_BULK_COMMUNICATION_OBJECT_HPP
#define INCLUDED_GHEX_BULK_COMMUNICATION_OBJECT_HPP

#include <memory>
#include <functional>
#include <vector>
#include <map>
#include <tuple>
#include <boost/mp11.hpp>
#include "./common/moved_bit.hpp"
#include "./common/utils.hpp"
#include "./communication_object_2.hpp"
#include "./rma/locality.hpp"
#include "./rma/range_factory.hpp"
#include "./rma/handle.hpp"

#include "./common/defs.hpp"
#ifdef GHEX_CUDACC
#include "./common/cuda_runtime.hpp"
#endif

namespace gridtools {
namespace ghex {

// RMA overview
// ============
//
// The GHEX direct memory access is implemented for threads/processes that share the same memory.
// This means, that in-node communication can be accelerated by by-passing message passing which
// is otherwise employed. In order for direct memory access to work, the memory must be prepared
// and the addresses must be communicated to the remote counter parts. Note, that word remote 
// signifies a process/thread on the same shared memory (in-node) which is writing into memory which
// it does not own (put operation). Conversly, the word local is used for a process/thread that
// owns a memory resource which it exposes for remote puts. Since GHEX does only use put operations
// (and no get operations), the local and remote processes/threads are also denoted as source and
// target: source is where the data comes from (remote) and target is where the data is put to
// (local).
//
// The rma facilites can be used for multi-threaded applications and multi-processed applications.
// In the case of multi-processed applications, the xpmem kernel module must be available for the
// transport to work.
//
// The multi-threaded parts are built on top of standard thread synchronization mechanisms, whereas
// the multi-processed parts use shmem for synchronization and xpmem for data exposure. GPUs are
// managed with cuda IPC facilities.
//
// The GHEX RMA facilities are comprised of several building blocks:
//
// 1. data handles: expose the data for remote read/write
// 2. access guards: synchronize access to data
// 3. events: additional synchronization mechanisms (used for GPUs)
// 4. ranges: abstract representation of a data portion which shall be exposed (i.e. a halo)
// 5. range factory: a class which type-erases fields/halos for transport through the network 
// 6. bulk communication object: user facing communication interface
// 7. range generator: type which can generate ranges from halos. This class needs to be implemented
// for each grid type in ghex. So far only regular grids are supported.
//


// type erased bulk communication object
// can be used for storing bulk communication objects
struct generic_bulk_communication_object
{
    struct bulk_co_iface;

    struct handle
    {
        std::function<void()> m_remote_wait_fct;
        bulk_co_iface* m_bulk_co_iface_ptr;
        void wait()
        {
            m_remote_wait_fct();
            m_bulk_co_iface_ptr->wait();
        }
    };

    struct bulk_co_iface
    {
        friend class handle;
        virtual ~bulk_co_iface() {}
        virtual handle exchange() = 0;
    protected:
        virtual void wait() = 0;
    };

    template<typename CO>
    struct bulk_co_impl : public bulk_co_iface
    {
        CO m;
        bulk_co_impl(CO&& co) : m{std::move(co)} {}
        handle exchange() override final
        {
            return {std::move(m.exchange().m_remote_handle.m_wait_fct), this };
        }

    private:
        void wait() override final { m.wait(); }
    };

private:
    std::unique_ptr<bulk_co_iface> m_impl;

public:
    generic_bulk_communication_object() = default;
    template<typename CO>
    generic_bulk_communication_object(CO&& co) : m_impl{ std::make_unique<bulk_co_impl<CO>>(std::move(co)) } {}
    generic_bulk_communication_object(generic_bulk_communication_object&&) = default;
    generic_bulk_communication_object& operator=(generic_bulk_communication_object&&) = default;

    handle exchange() { return m_impl->exchange(); }
};

/** @brief Communication object which enables registration of fields ahead of communication so that
  * halo exchange operation can be called repeatedly. This class also enables direct memory access
  * when possible to the neighboring fields (RMA). Current RMA is limited to in-node threads and
  * processes. Inter-process RMA is only enabled when GHEX is built with xpmem support.
  * @tparam RangeGen template template parameter which generates source and target ranges
  * @tparam Pattern the pattern type that can be used with the registered fields
  * @tparam Fields a list of field types that can be registered */
template<template <typename> class RangeGen, typename Pattern, typename... Fields>
class bulk_communication_object
{
public: // member types
    using pattern_type = Pattern;
    using communicator_type = typename pattern_type::communicator_type;
    using grid_type = typename pattern_type::grid_type;
    using domain_id_type = typename pattern_type::domain_id_type;
    using co_type = communication_object<communicator_type,grid_type,domain_id_type>;
    using co_handle = typename co_type::handle_type;
    template<typename Field>
    using buffer_info_type = typename co_type::template buffer_info_type<typename Field::arch_type, Field>;

    friend class generic_bulk_communication_object::bulk_co_impl<bulk_communication_object>;

    // return type from exchange()
    struct handle
    {
        co_handle m_remote_handle;
        bulk_communication_object* m_bulk_co_ptr;

        void wait()
        {
            m_remote_handle.wait();
            m_bulk_co_ptr->wait();
        }
    };

private: // member types
    // this type holds the patterns used for remote and local exchanges
    // map key is the pointer to the pattern that is used when the field is added
    using pattern_map = std::map<const pattern_type*, pattern_type>;

    // a similar map that holds rma handles to each field that is added
    // map key is the pointer to the fields memory
    using local_handle_map = std::map<void*, rma::local_handle>;

    // template meta function: changes a field type's architecture
    template<typename A>
    struct select_arch_q
    {
        template<typename Field>
        using fn = typename Field::template rebind_arch<A>;
    };

    // get the actual range type that is used from the range generator
    template<typename Field>
    using select_range = typename RangeGen<Field>::range_type;
    
    // unique list of field types
    using field_types = boost::mp11::mp_unique<boost::mp11::mp_list<Fields...>>;
    // buffer info types from this list
    using buffer_info_types = boost::mp11::mp_transform<buffer_info_type, field_types>;
    // all possible cpu field types
    using cpu_fields = boost::mp11::mp_transform_q<select_arch_q<cpu>,field_types>;
#ifdef GHEX_CUDACC
    // all possible gpu field types
    using gpu_fields = boost::mp11::mp_transform_q<select_arch_q<gpu>,field_types>;
    using all_fields = boost::mp11::mp_unique<boost::mp11::mp_append<cpu_fields,gpu_fields>>;
#else
    using all_fields = boost::mp11::mp_unique<cpu_fields>;
#endif
    // the range types form all the possible field types
    using all_ranges = boost::mp11::mp_transform<select_range,all_fields>;
    // the range factory type
    using range_factory = rma::range_factory<all_ranges>;

    template<typename Field>
    using select_target_range = std::vector<typename RangeGen<Field>::template target_range<
        range_factory,communicator_type>>;
    template<typename Field>
    using select_source_range = std::vector<typename RangeGen<Field>::template source_range<
        range_factory,communicator_type>>;

    // generated target range type
    template<typename Field>
    struct target_ranges
    {
        using ranges_type = select_target_range<Field>;
        using range_type = typename ranges_type::value_type;
        std::vector<ranges_type> m_ranges;
    };

    // generated source range type
    template<typename Field>
    struct source_ranges
    {
        using ranges_type = select_source_range<Field>;
        using range_type = typename ranges_type::value_type;
        std::vector<ranges_type> m_ranges; 
    };

    // class that holds a field, it's rma handle, and local and remote patterns
    template<typename Field>
    struct field_container
    {
        Field m_field;
        rma::local_handle& m_local_handle;
        pattern_type& m_remote_pattern;
        pattern_type& m_local_pattern;

        // construct from field, pattern and the maps storing rma handles, and patterns
        field_container(communicator_type comm, const Field& f, const pattern_type& pattern,
            local_handle_map& l_handle_map, pattern_map& local_map, pattern_map& remote_map)
        : m_field{f}
        , m_local_handle(l_handle_map.insert(std::make_pair((void*)(f.data()),rma::local_handle{})).first->second)
        , m_remote_pattern(remote_map.insert(std::make_pair(&pattern, pattern)).first->second)
        , m_local_pattern(local_map.insert(std::make_pair(&pattern, pattern)).first->second)
        {
            // initialize the remote handle - this will effectively publish the rma pointers
            // will do nothing if already initialized
            m_local_handle.init(f.data(), f.bytes(), std::is_same<typename Field::arch_type, gpu>::value);

            // prepare local and remote patterns
            // =================================

            // loop over all subdomains in pattern
            for (int n = 0; n<pattern.size(); ++n)
            {
                // remove local fields from remote pattern
                auto& r_p = m_remote_pattern[n];
                auto r_it = r_p.send_halos().begin();
                while (r_it != r_p.send_halos().end())
                {
                    const auto local = rma::is_local(comm, r_it->first.mpi_rank);
                    if (local != rma::locality::remote) r_it = r_p.send_halos().erase(r_it);
                    else ++r_it;
                }

                // remove remote fields from local pattern
                auto& l_p = m_local_pattern[n];
                auto l_it = l_p.send_halos().begin();
                while (l_it != l_p.send_halos().end())
                {
                    const auto local = rma::is_local(comm, l_it->first.mpi_rank);
                    if (local != rma::locality::remote) ++l_it;
                    else l_it = l_p.send_halos().erase(l_it);
                }

                // remove local fields from remote pattern
                r_it = r_p.recv_halos().begin();
                while (r_it != r_p.recv_halos().end())
                {
                    const auto local = rma::is_local(comm, r_it->first.mpi_rank);
                    if (local != rma::locality::remote) r_it = r_p.recv_halos().erase(r_it);
                    else ++r_it;
                }

                // remove remote fields from local pattern
                l_it = l_p.recv_halos().begin();
                while (l_it != l_p.recv_halos().end())
                {
                    const auto local = rma::is_local(comm, l_it->first.mpi_rank);
                    if (local != rma::locality::remote) ++l_it;
                    else l_it = l_p.recv_halos().erase(l_it);
                }
            }
        }

        field_container(const field_container&) = default;
        field_container(field_container&&) = default;
    };

    template<typename Field>
    using field_vector = std::vector<field_container<Field>>;
    
    // a tuple of field_vectors which are vectors of field_containers
    using field_container_t = boost::mp11::mp_rename<boost::mp11::mp_transform<field_vector,field_types>,std::tuple>;
    
    // a tuple of buffer infos: tuple of vectors of buffer_info_types
    using buffer_info_container_t = boost::mp11::mp_rename<boost::mp11::mp_transform<std::vector,buffer_info_types>,std::tuple>;

    // source and target range tuples: tuple of target_ranges/source_ranges
    using target_ranges_t = boost::mp11::mp_rename<boost::mp11::mp_transform<target_ranges, field_types>,std::tuple>;
    using source_ranges_t = boost::mp11::mp_rename<boost::mp11::mp_transform<source_ranges, field_types>,std::tuple>;

    struct co_deleter
    {
        bool m_owning;
        void operator()(co_type* ptr) const { if (m_owning) delete ptr; }
    };

    using co_ptr = std::unique_ptr<co_type, co_deleter>;

    // struct holding a function which implements a request
    struct func_request
    {
        std::function<bool()> m_fct;

        func_request(std::function<bool()>&& fct) : m_fct(std::move(fct)) {}
        func_request(const func_request&) = delete;
        func_request(func_request&&) = default;

        bool test() noexcept { return m_fct(); }
    };

private: // members
    communicator_type                  m_comm;
    co_ptr                             m_co;
    pattern_map                        m_local_pattern_map;
    pattern_map                        m_remote_pattern_map;
    field_container_t                  m_field_container_tuple;
    buffer_info_container_t            m_buffer_info_container_tuple;
    target_ranges_t                    m_target_ranges_tuple;
    source_ranges_t                    m_source_ranges_tuple;
    local_handle_map                   m_local_handle_map;
    moved_bit                          m_moved;
    bool                               m_initialized = false;
    std::vector<func_request>          m_put_funcs;
    std::vector<func_request>          m_wait_funcs;
    std::vector<std::function<void()>> m_open_funcs;
#ifdef GHEX_BULK_UNIQUE_TAGS
    std::map<int,int>                  m_tag_map;
#endif

public: // ctors
    bulk_communication_object(communicator_type comm)
    : m_comm(comm)
    , m_co{new co_type(comm), co_deleter{true}}
    {}

    bulk_communication_object(co_type& co)
    : m_comm(co.communicator())
    , m_co{&co, co_deleter{false}}
    {}

    // move only
    bulk_communication_object(const bulk_communication_object&) = delete;
    bulk_communication_object(bulk_communication_object&&) = default;

public:
    /** @brief add a field in the usual manner, f.ex. add_field(my_pattern(my_field)).
      * @tparam Field field type
      * @param bi buffer info object generated from pattern and field */
    template<typename Field>
    void add_field(buffer_info_type<Field> bi)
    {
        if (m_moved)
            throw std::runtime_error("error: trying to add a field to a CO which is moved");
        if (m_initialized)
            throw std::runtime_error("error: this CO has been initialized already");

        using f_cont_t  = std::vector<field_container<Field>>;
        using t_range_t = target_ranges<Field>;
        using s_range_t = source_ranges<Field>;

        auto& f_cont  = std::get<f_cont_t>(m_field_container_tuple);
        auto& t_range = std::get<t_range_t>(m_target_ranges_tuple);
        auto& s_range = std::get<s_range_t>(m_source_ranges_tuple);

        // store field
        f_cont.push_back(field_container<Field>(m_comm, bi.get_field(), bi.get_pattern_container(),
            m_local_handle_map, m_local_pattern_map, m_remote_pattern_map));
        s_range.m_ranges.resize(s_range.m_ranges.size()+1);
        t_range.m_ranges.resize(t_range.m_ranges.size()+1);

        auto& f = f_cont.back().m_field;
        auto field_info = f_cont.back().m_local_handle.get_info();
        // loop over patterns 
        for (auto& p : f_cont.back().m_local_pattern)
        {
            // check if field has the right domain
            if (f.domain_id() == p.domain_id())
            {
#ifdef GHEX_BULK_UNIQUE_TAGS
                auto m_it = m_tag_map.insert(std::pair<int,int>(f.domain_id(),0)).first;
#endif
                // loop over halos and set up source ranges
                for (auto h_it = p.send_halos().begin(); h_it != p.send_halos().end(); ++h_it)
                {
                    int q=0;
                    for (auto it = h_it->second.rbegin(); it != h_it->second.rend(); ++it)
                    {
                        const auto& c = *it;
                        s_range.m_ranges.back().emplace_back(
                            m_comm, f, c, h_it->first.mpi_rank
#ifdef GHEX_BULK_UNIQUE_TAGS
                            , (m_it->second + h_it->first.tag+1)*10000 + q
#else
                            // alternatively rely on message ordering:
                            , h_it->first.tag
#endif
                            ); 
                        ++q;
                    }
                }
                // loop over halos and set up target
                for (auto h_it = p.recv_halos().begin(); h_it != p.recv_halos().end(); ++h_it)
                {
                    int q=0;
                    for (auto it = h_it->second.rbegin(); it != h_it->second.rend(); ++it)
                    {
                        const auto local = rma::is_local(m_comm, h_it->first.mpi_rank);
                        const auto& c = *it;
                        t_range.m_ranges.back().emplace_back(
                            m_comm, f, field_info, c, h_it->first.mpi_rank
#ifdef GHEX_BULK_UNIQUE_TAGS
                            , (m_it->second + h_it->first.tag+1)*10000 + q
#else
                            // alternatively rely on message ordering:
                            , h_it->first.tag
#endif
                            , local); 
                        ++q;
                    }
                }
#ifdef GHEX_BULK_UNIQUE_TAGS
                m_it->second += (f_cont.back().m_local_pattern.max_tag()+1);
#endif
            }
        }
    }

    // is co initialized
    bool initialized()
    {
        return m_initialized;
    }
    
    // add multiple fields at once
    template<typename... F>
    void add_fields(buffer_info_type<F>... bis)
    {
        auto bis_tp = std::make_tuple(bis...);
        for (std::size_t i=0; i<sizeof...(F); ++i)
        {
            boost::mp11::mp_with_index<sizeof...(F)>(i,
            [this,&bis_tp](auto i) {
                // get the field Index 
                using I = decltype(i);
                add_field(std::get<I::value>(bis_tp));
            });
        }
    }

    /** @brief initializes the bulk communication object after which point no more fields can be 
      * added. A call to this function will also make sure that all RMA handles are exchanged.
      * It is not necessary to call this function manually since it is called in the first exchange.
      * Calls to init may become necessary when different bulk communication objects are used at the
      * same time. */
    void init()
    {
        if (m_initialized) return;

        // loop over Fields
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get source and target ranges
                auto& bi_cont = std::get<I::value>(m_buffer_info_container_tuple);
                auto& f_cont  = std::get<I::value>(m_field_container_tuple);
                // add remote exchange
                for (auto& f : f_cont)
                    bi_cont.push_back( f.m_remote_pattern(f.m_field) );
            });
        }
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get source and target ranges
                auto& s_range = std::get<I::value>(m_source_ranges_tuple);
                // complete the handshake
                for (auto& s_vec : s_range.m_ranges)
                    for (auto& r : s_vec)
                        r.recv();
            });
        }
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get source and target ranges
                auto& t_range = std::get<I::value>(m_target_ranges_tuple);
                // complete the handshake
                for (auto& t_vec : t_range.m_ranges)
                    for (auto& r : t_vec)
                        r.send();
            });
        }

        // loop over Fields
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                
                // get target ranges for fields
                auto& t_range = std::get<I::value>(m_target_ranges_tuple);
                for (auto& t_vec : t_range.m_ranges)
                    for (auto& r : t_vec)
                    {
                        // register open functions
                        m_open_funcs.push_back([&r](){ r.end_target_epoch(); });
                        // register wait functions
                        m_wait_funcs.push_back(func_request{std::function<bool()>([&r]() -> bool
                            { return r.try_start_target_epoch(); })});
                    }
                
                // get source ranges for fields
                auto& s_range = std::get<I::value>(m_source_ranges_tuple);
                // put data
                for (auto& s_vec : s_range.m_ranges)
                    for (auto& r : s_vec)
                    {
                        // register put functions
                        m_put_funcs.push_back(func_request{std::function<bool()>([&r]() -> bool
                            {
                                if (r.try_start_source_epoch())
                                {
                                    r.put();
                                    r.end_source_epoch();
                                    return true;
                                }
                                else return false;
                            })});
                    }
            });
        }
        m_initialized = true;
    }
    
private: // helper functions to handle the remote exchanges
    co_handle exchange_remote()
    {
        return exchange_remote(std::make_index_sequence<boost::mp11::mp_size<field_types>::value>());
    }

    template<std::size_t... I>
    co_handle exchange_remote(std::index_sequence<I...>)
    {
        return m_co->exchange(std::make_pair(std::get<I>(m_buffer_info_container_tuple).begin(),
            std::get<I>(m_buffer_info_container_tuple).end())...);
    }

public:
    /** @brief do an exchange of halos
      * @return handle to wait on (for the remote exchanges) */
    handle exchange()
    {
        if (!m_initialized) init();
        // loop over Fields for making the ranges writable for remotes
        for (auto& x : m_open_funcs) x();
        // start remote exchange
        auto h = exchange_remote();
        // put data as soon as ranges are writable
        await_requests(m_put_funcs, [comm = m_comm]() mutable {comm.progress();});
        return {std::move(h),this};
    }

private:
    void wait()
    {
        // wait for all local ranges to be filled
        await_requests(m_wait_funcs);
    }
};

} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_BULK_COMMUNICATION_OBJECT_HPP */
