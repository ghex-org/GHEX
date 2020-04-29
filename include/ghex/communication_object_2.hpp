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
#ifndef INCLUDED_GHEX_COMMUNICATION_OBJECT_2_HPP
#define INCLUDED_GHEX_COMMUNICATION_OBJECT_2_HPP

#include "./cuda_utils/stream.hpp"
#include "./packer.hpp"
#include "./common/utils.hpp"
#include "./common/test_eq.hpp"
#include "./buffer_info.hpp"
#include "./transport_layer/tags.hpp"
#include "./arch_traits.hpp"
#include <map>
#include <stdio.h>
#include <functional>

namespace gridtools {

    namespace ghex {

        // forward declaration
        template<typename Communicator, typename GridType, typename DomainIdType>
        class communication_object;

        /** @brief handle type for waiting on asynchronous communication processes.
          * The wait function is stored in a member.
          * @tparam Transport message transport type
          * @tparam GridType grid tag type
          * @tparam DomainIdType domain id type*/
        template<typename Communicator, typename GridType, typename DomainIdType>
        class communication_handle
        {
        private: // friend class

            friend class communication_object<Communicator,GridType,DomainIdType>;

        private: // member types

            using co_t              = communication_object<Communicator,GridType,DomainIdType>;
            using communicator_type = Communicator;

        private: // members

            communicator_type m_comm;
            std::function<void()> m_wait_fct;

        public: // public constructor

            /** @brief construct a ready handle
              * @param comm communicator */
            communication_handle(const communicator_type& comm) 
            : m_comm{comm} {}

        private: // private constructor

            /** @brief construct a handle with a wait function
              * @tparam Func function type with signature void()
              * @param comm communicator
              * @param wait_fct wait function */
            template<typename Func>
            communication_handle(const communicator_type& comm, Func&& wait_fct) 
            : m_comm{comm}, m_wait_fct(std::forward<Func>(wait_fct)) {}

        public: // copy and move ctors

            communication_handle(communication_handle&&) = default;
            communication_handle(const communication_handle&) = delete;
            communication_handle& operator=(communication_handle&&) = default;
            communication_handle& operator=(const communication_handle&) = delete;

        public: // member functions

            /** @brief  wait for communication to be finished*/
            void wait() { if (m_wait_fct) m_wait_fct(); }
        };

     
        /** @brief communication object responsible for exchanging halo data. Allocates storage depending on the 
         * device type and device id of involved fields.
         * @tparam Transport message transport type
         * @tparam GridType grid tag type
         * @tparam DomainIdType domain id type*/
        template<typename Communicator, typename GridType, typename DomainIdType>
        class communication_object
        {
        private: // friend class

            friend class communication_handle<Communicator,GridType,DomainIdType>;

        public: // member types

            /** @brief handle type returned by exhange operation */
            using handle_type             = communication_handle<Communicator,GridType,DomainIdType>;
            //using transport_type          = Transport;
            using grid_type               = GridType;
            using domain_id_type          = DomainIdType;
            using pattern_type            = pattern<Communicator,GridType,DomainIdType>;
            using pattern_container_type  = pattern_container<Communicator,GridType,DomainIdType>;
            using this_type               = communication_object<Communicator,GridType,DomainIdType>;

            template<typename D, typename F>
            using buffer_info_type        = buffer_info<pattern_type,D,F>;

        private: // member types

            using communicator_type       = Communicator; //typename handle_type::communicator_type;
            using address_type            = typename communicator_type::address_type;
            using index_container_type    = typename pattern_type::index_container_type;
            using pack_function_type      = std::function<void(void*,const index_container_type&, void*)>;
            using unpack_function_type    = std::function<void(const void*,const index_container_type&, void*)>;

            /** @brief pair of domain ids with ordering */
            struct domain_id_pair
            {
                domain_id_type first_id;
                domain_id_type second_id;
                bool operator<(const domain_id_pair& other) const noexcept
                {
                    return (first_id < other.first_id ? true : 
                            (first_id > other.first_id ? false : (second_id < other.second_id)));
                }
            };

            /** @brief Holds a pointer to a set of iteration spaces and a callback function pointer 
              * which is used to store a field's pack or unpack member function. 
              * This class also stores the offset in the serialized buffer in bytes.
              * The type-erased field_ptr member is only used for the gpu-vector-interface.
              * @tparam Function Either pack or unpack function pointer type */
            template<typename Function>
            struct field_info
            {
                using index_container_type = typename pattern_type::map_type::mapped_type;
                Function call_back;
                const index_container_type* index_container;
                std::size_t offset;
                void* field_ptr;
            };

            /** @brief Holds serial buffer memory and meta information associated with it
              * @tparam Vector contiguous buffer memory type
              * @tparam Function Either pack or unpack function pointer type */
            template<class Vector, class Function>
            struct buffer
            {
                using field_info_type = field_info<Function>;
                address_type address;
                int tag;
                Vector buffer;
                std::size_t size;
                std::vector<field_info_type> field_infos;
                cuda::stream m_cuda_stream;
            };

            /** @brief Holds maps of buffers for send and recieve operations indexed by a domain_id_pair and a device id
              * @tparam Arch the device on which the buffer memory is allocated */
            template<typename Arch>
            struct buffer_memory
            {
                using arch_type        = Arch;
                using device_id_type   = typename arch_traits<Arch>::device_id_type;
                using vector_type      = typename arch_traits<Arch>::message_type;
                
                using send_buffer_type = buffer<vector_type,pack_function_type>; 
                using recv_buffer_type = buffer<vector_type,unpack_function_type>; 
                using send_memory_type = std::map<device_id_type, std::map<domain_id_pair,send_buffer_type>>;
                using recv_memory_type = std::map<device_id_type, std::map<domain_id_pair,recv_buffer_type>>;

                std::map<device_id_type, std::unique_ptr<typename arch_traits<Arch>::pool_type>> m_pools;
                send_memory_type send_memory;
                recv_memory_type recv_memory;

                // additional members needed for receive operations used for scheduling calls to unpack
                using hook_type       = recv_buffer_type*;
                using future_type     = typename communicator_type::template future<hook_type>;
                std::vector<future_type> m_recv_futures;

            };
            
            /** tuple type of buffer_memory (one element for each device in arch_list) */
            using memory_type = detail::transform<arch_list>::with<buffer_memory>;

        private: // members

            bool m_valid;
            communicator_type m_comm;
            memory_type m_mem;
            std::vector<typename communicator_type::template future<void>> m_send_futures;

        public: // ctors

            communication_object(communicator_type comm)
            : m_valid(false) 
            , m_comm(comm)
            {}
            communication_object(const communication_object&) = delete;
            communication_object(communication_object&&) = default;

        public: // exchange arbitrary field-device-pattern combinations

            /** @brief blocking variant of halo exchange
              * @tparam Archs list of device types
              * @tparam Fields list of field types
              * @param buffer_infos buffer_info objects created by binding a field descriptor to a pattern */
            template<typename... Archs, typename... Fields>
            void bexchange(buffer_info_type<Archs,Fields>... buffer_infos)
            {
                exchange(buffer_infos...).wait();
            }

            /** @brief non-blocking exchange of halo data
              * @tparam Archs list of device types
              * @tparam Fields list of field types
              * @param buffer_infos buffer_info objects created by binding a field descriptor to a pattern
              * @return handle to await communication */
            template<typename... Archs, typename... Fields>
            [[nodiscard]] handle_type exchange(buffer_info_type<Archs,Fields>... buffer_infos)
            {
                // check that arguments are compatible
                using test_t = pattern_container<communicator_type,grid_type,domain_id_type>;
                static_assert(detail::test_eq_t<test_t, typename buffer_info_type<Archs,Fields>::pattern_container_type...>::value,
                        "patterns are not compatible with this communication object");
                if (m_valid) 
                    throw std::runtime_error("earlier exchange operation was not finished");
                m_valid = true;

                // temporarily store address of pattern containers
                const test_t* ptrs[sizeof...(Fields)] = { &(buffer_infos.get_pattern_container())... };
                // build a tag map
                std::map<const test_t*,int> pat_ptr_map;
                int max_tag = 0;
                for (unsigned int k=0; k<sizeof...(Fields); ++k)
                {
                    auto p_it_bool = pat_ptr_map.insert( std::make_pair(ptrs[k], max_tag) );
                    if (p_it_bool.second == true)
                        max_tag += ptrs[k]->max_tag()+1;
                }
                // compute tag offset for each field
                int tag_offsets[sizeof...(Fields)] = { pat_ptr_map[&(buffer_infos.get_pattern_container())]... };
                // store arguments and corresponding memory in tuples
                using buffer_infos_ptr_t     = std::tuple<std::remove_reference_t<decltype(buffer_infos)>*...>;
                using memory_t               = std::tuple<buffer_memory<Archs>*...>;
                buffer_infos_ptr_t buffer_info_tuple{&buffer_infos...};
                memory_t memory_tuple{&(std::get<buffer_memory<Archs>>(m_mem))...};
                // loop over buffer_infos/memory and compute required space
                int i = 0;
                detail::for_each(memory_tuple, buffer_info_tuple, [this,&i,&tag_offsets](auto mem, auto bi) 
                {
                    using arch_type = typename std::remove_reference_t<decltype(*mem)>::arch_type;
                    using value_type  = typename std::remove_reference_t<decltype(*bi)>::value_type;
                    auto field_ptr = &(bi->get_field());
                    const domain_id_type my_dom_id = bi->get_field().domain_id();
                    allocate<arch_type,value_type>(mem, bi->get_pattern(), field_ptr, my_dom_id, bi->device_id(), tag_offsets[i]);
                    ++i;
                });
                handle_type h(m_comm, [this](){this->wait();});
                post_recvs();
                pack();
                return h; 
            }

//        public: // exchange a number of buffer_infos with Field = field_descriptor (optimization for gpu below)
//
//#ifdef __CUDACC__
//            template<typename Arch, typename T, int... Order>
//            [[nodiscard]] std::enable_if_t<std::is_same<Arch,gpu>::value, handle_type>
//            exchange_u(
//                buffer_info_type<Arch,structured::regular::field_descriptor<T,Arch,structured::regular::domain_descriptor<domain_id_type,sizeof...(Order)>,Order...>>* first, 
//                std::size_t length)
//            {
//                using memory_t   = buffer_memory<gpu>;
//                using field_type = std::remove_reference_t<decltype(first->get_field())>;
//                using value_type = typename field_type::value_type;
//                auto h = exchange_impl(first, length);
//                post_recvs();
//                h.m_wait_fct = [this](){this->wait_u<value_type,field_type>();};
//                memory_t& mem = std::get<memory_t>(m_mem);
//                packer<gpu>::template pack_u<value_type,field_type>(mem, m_send_futures, m_comm);
//                return h;
//            }
//#endif

            template<typename... Iterators>
            [[nodiscard]]
            handle_type exchange(Iterators... iters)
            {
                static_assert(sizeof...(Iterators) % 2 == 0, "need even number of iteratiors: (begin,end) pairs");
                return exchange(std::make_index_sequence<sizeof...(iters)/2>(), iters...); 
            }

        private: // implementation
            template<typename Tuple, typename... Iterators>
            [[nodiscard]]
            handle_type exchange(std::pair<Iterators,Iterators>... iter_pairs)
            {
                using pattern_container_types =
                    std::tuple<typename std::remove_reference<decltype(*(iter_pairs.first))>::type::
                        pattern_container_type...>;
                static_assert(std::is_same<Tuple,pattern_container_types>::value,
                    "patterns are incompatible with this communication object" );

                const std::tuple<std::pair<Iterators,Iterators>...> iter_pairs_t{iter_pairs...};

                if (m_valid)
                    throw std::runtime_error("earlier exchange operation was not finished");
                m_valid = true;

                // build a tag map
                using test_t = pattern_container<communicator_type,grid_type,domain_id_type>;
                std::map<const test_t*,int> pat_ptr_map;
                int max_tag = 0;
                detail::for_each(iter_pairs_t, [&pat_ptr_map,&max_tag](auto iter_pair) {
                    for (auto it=iter_pair.first; it!=iter_pair.second; ++it) {
                        auto ptr = &(it->get_pattern_container());
                        auto p_it_bool = pat_ptr_map.insert( std::make_pair(ptr, max_tag) );
                        if (p_it_bool.second == true)
                            max_tag += ptr->max_tag()+1;
                    }
                });
                detail::for_each(iter_pairs_t, [this,&pat_ptr_map](auto iter_pair) {
                    using buffer_info_t = typename std::remove_reference<decltype(*iter_pair.first)>::type;
                    using arch_t = typename buffer_info_t::arch_type;
                    using value_t = typename buffer_info_t::value_type;
                    auto mem = &(std::get<buffer_memory<arch_t>>(m_mem));
                    for (auto it=iter_pair.first; it!=iter_pair.second; ++it) {
                        auto field_ptr = &(it->get_field());
                        auto tag_offset = pat_ptr_map[ &(it->get_pattern_container()) ];
                        const auto my_dom_id = it->get_field().domain_id();
                        allocate<arch_t,value_t>(mem, it->get_pattern(), field_ptr, my_dom_id,
                            it->device_id(), tag_offset);
                    }
                });

                post_recvs();
                pack();
                return handle_type(m_comm, [this](){this->wait();});
            }
            
            template<std::size_t... Is, typename... Iterators>
            [[nodiscard]]
            handle_type exchange(std::index_sequence<Is...>, Iterators... iters)
            {
                const std::tuple<Iterators...> iter_t{iters...};
                using test_t = std::tuple<pattern_container<communicator_type,grid_type,domain_id_type>>;
                using test_t_n = std::tuple< typename std::tuple_element<Is*0, test_t>::type... >;
                return exchange<test_t_n>(std::make_pair(std::get<2*Is>(iter_t), std::get<2*Is+1>(iter_t))...);
            }

            void post_recvs()
            {
                detail::for_each(m_mem, [this](auto& m)
                {
                    for (auto& p0 : m.recv_memory)
                    {
                        for (auto& p1: p0.second)
                        {
                            if (p1.second.size > 0u)
                            {
                                p1.second.buffer.resize(p1.second.size);
                                m.m_recv_futures.emplace_back(
                                    typename std::remove_reference_t<decltype(m)>::future_type{
                                        &p1.second,
                                        m_comm.recv(p1.second.buffer, p1.second.address, p1.second.tag).m_handle});
                            }
                        }
                    }
                });
            }

            void pack()
            {
                detail::for_each(m_mem, [this](auto& m)
                {
                    using arch_type = typename std::remove_reference_t<decltype(m)>::arch_type;
                    packer<arch_type>::pack(m,m_send_futures,m_comm);
                });
            }

        private: // wait functions

            void wait()
            {
                if (!m_valid) return;
                detail::for_each(m_mem, [this](auto& m)
                {
                    using arch_type = typename std::remove_reference_t<decltype(m)>::arch_type;
                    packer<arch_type>::unpack(m);
                });
                for (auto& f : m_send_futures) 
                    f.wait();
                clear();
            }

#ifdef __CUDACC__
            template<typename T, typename Field>
            void wait_u()
            {
                if (!m_valid) return;
                using memory_t   = buffer_memory<gpu>;
                memory_t& mem = std::get<memory_t>(m_mem);
                packer<gpu>::template unpack_u<T,Field>(mem);
                for (auto& f : m_send_futures) 
                    f.wait();
                clear();
            }
#endif
        
        private: // reset

            // clear the internal flags so that a new exchange can be started
            // important: does not deallocate
            void clear()
            {
                m_valid = false;
                m_send_futures.clear();
                detail::for_each(m_mem, [this](auto& m)
                {
                    m.m_recv_futures.clear();
                    for (auto& p0 : m.send_memory)
                        for (auto& p1 : p0.second)
                        {
                            p1.second.buffer.resize(0);
                            p1.second.size = 0;
                            p1.second.field_infos.resize(0);
                        }
                    for (auto& p0 : m.recv_memory)
                        for (auto& p1 : p0.second)
                        {
                            p1.second.buffer.resize(0);
                            p1.second.size = 0;
                            p1.second.field_infos.resize(0);
                        }
                });
            }

        private: // allocation member functions

            template<typename Arch, typename T, typename Memory, typename Field, typename O>
            void allocate(Memory& mem, const pattern_type& pattern, Field* field_ptr, domain_id_type dom_id, typename arch_traits<Arch>::device_id_type device_id, O tag_offset)
            {
                auto& pool = mem->m_pools[device_id];
                if (!pool)
                {
                    pool.reset( new typename arch_traits<Arch>::pool_type{ typename arch_traits<Arch>::basic_allocator_type{} } );
                }
                allocate<Arch,T,typename buffer_memory<Arch>::recv_buffer_type>( 
                    mem->recv_memory[device_id], 
                    pattern.recv_halos(),
                    [field_ptr](const void* buffer, const index_container_type& c, void* arg) 
                    {
                        field_ptr->unpack(reinterpret_cast<const T*>(buffer),c,arg); 
                    },
                    dom_id, 
                    device_id, 
                    tag_offset, 
                    true, 
                    *pool,
                    field_ptr);
                allocate<Arch,T,typename buffer_memory<Arch>::send_buffer_type>(
                    mem->send_memory[device_id], 
                    pattern.send_halos(),
                    [field_ptr](void* buffer, const index_container_type& c, void* arg) 
                    {
                        field_ptr->pack(reinterpret_cast<T*>(buffer),c,arg);
                    },
                    dom_id, 
                    device_id, 
                    tag_offset, 
                    false, 
                    *pool, 
                    field_ptr);
            }

            // compute memory requirements to be allocated on the device
            template<typename Arch, typename ValueType, typename BufferType, typename Memory, typename Halos, typename Function, typename DeviceIdType, 
                typename Pool, typename Field = void>
            void allocate(Memory& memory, const Halos& halos, Function&& func, domain_id_type my_dom_id, DeviceIdType device_id, 
                          int tag_offset, bool receive, Pool& pool, Field* field_ptr = nullptr)
            {
                for (const auto& p_id_c : halos)
                {
                    const auto num_elements = pattern_type::num_elements(p_id_c.second)*
                        field_ptr->num_components();
                    if (num_elements < 1) continue;
                    const auto remote_address = p_id_c.first.address;
                    const auto remote_dom_id  = p_id_c.first.id;
                    domain_id_type left, right;
                    if (receive) 
                    {
                        left  = my_dom_id;
                        right = remote_dom_id;
                    }
                    else
                    {
                        left  = remote_dom_id;
                        right = my_dom_id;
                    }
                    const auto d_p = domain_id_pair{left,right};
                    auto it = memory.find(d_p);
                    if (it == memory.end())
                    {
                        it = memory.insert(std::make_pair(
                            d_p,
                            BufferType{
                                remote_address,
                                p_id_c.first.tag+tag_offset,
                                arch_traits<Arch>::make_message(pool, device_id),
                                0,
                                std::vector<typename BufferType::field_info_type>(),
                                cuda::stream()
                            })).first;
                    }
                    else if (it->second.size==0)
                    {
                        it->second.address = remote_address;
                        it->second.tag = p_id_c.first.tag+tag_offset;
                        it->second.field_infos.resize(0);
                    }
                    const auto prev_size = it->second.size;
                    const auto padding = ((prev_size+alignof(ValueType)-1)/alignof(ValueType))*alignof(ValueType) - prev_size;
                    it->second.field_infos.push_back(
                        typename BufferType::field_info_type{std::forward<Function>(func), &p_id_c.second, prev_size + padding, field_ptr});
                    it->second.size += padding + static_cast<std::size_t>(num_elements)*sizeof(ValueType);
                }
            }
        };

        /** @brief creates a communication object based on the pattern type
          * @tparam PatternContainer pattern type
          * @return communication object */
        template<typename PatternContainer>
        auto make_communication_object(typename PatternContainer::value_type::communicator_type comm)
        {
            //using transport_type   = typename PatternContainer::value_type::communicator_type::transport_type;
            using communicator_type = typename PatternContainer::value_type::communicator_type;
            using grid_type         = typename PatternContainer::value_type::grid_type;
            using domain_id_type    = typename PatternContainer::value_type::domain_id_type;
            return communication_object<communicator_type,grid_type,domain_id_type>(comm);
        }

    } // namespace ghex
        
} // namespace gridtools

#endif /* INCLUDED_GHEX_COMMUNICATION_OBJECT_2_HPP */

