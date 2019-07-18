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
#ifndef INCLUDED_COMMUNICATION_OBJECT_2_HPP
#define INCLUDED_COMMUNICATION_OBJECT_2_HPP

#include "./utils.hpp"
#include "./buffer_info.hpp"
#include "./devices.hpp"
#include "./protocol/communicator_base.hpp"
#include <thread>
#include <future>

namespace gridtools {

    namespace detail {

        // forward declaration
        template<typename Tuple>
        struct transform;

        // transform a tuple of types into another tuple of types
        template<template<typename...> class Tuple, typename... Ts>
        struct transform<Tuple<Ts...>>
        {
            // by applying the compile-time transform CT
            template<template<typename> class CT>
            using with = Tuple<CT<Ts>...>;
        };

    } // namespace detail

    // forward declaration
    template<typename P, typename GridType, typename DomainIdType>
    class communication_object;

    /** @brief handle type for waiting on asynchronous communication processes
     * @tparam P message protocol type
     * @tparam GridType grid tag type
     * @tparam DomainIdType domain id type*/
    template<typename P, typename GridType, typename DomainIdType>
    class communication_handle
    {
    private: // friend class
        friend class communication_object<P,GridType,DomainIdType>;

    private: // member types
        using co_t              = communication_object<P,GridType,DomainIdType>;
        using communicator_type = protocol::communicator<P>;

    private: // private constructor
        communication_handle(co_t& co, const communicator_type& comm) 
        : m_co{&co}, m_comm{comm} {}

    public: // copy and move ctors
        communication_handle(communication_handle&&) = default;
        communication_handle(const communication_handle&) = delete;

    public: // member functions
        /** @brief  wait for communication to be finished*/
        void wait() { m_co->wait(); }
#ifdef GHEX_COMM_2_TIMINGS
        template<typename Timings>
        void wait(Timings& t) { m_co->wait(t); }
#endif

    private: // members
        co_t* m_co;
        communicator_type m_comm;
    };


    template<typename Device>
    struct packer
    {
        template<typename Map, typename Futures, typename Communicator>
        static void pack(Map& map, Futures& send_futures,Communicator& comm)
        {
            for (auto& p0 : map.send_memory)
            {
                for (auto& p1: p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        p1.second.buffer.resize(p1.second.size);
                        for (const auto& fb : p1.second.field_buffers)
                            fb.call_back( p1.second.buffer.data() + fb.offset, *fb.index_container, (void*)0);
                        //std::cout << "isend(" << p1.second.address << ", " << p1.second.tag 
                        //<< ", " << p1.second.buffer.size() << ")" << std::endl;
                        send_futures.push_back(comm.isend(
                            p1.second.address,
                            p1.second.tag,
                            p1.second.buffer));
                    }
                }
            }
        }

#ifndef GHEX_COMM_2_TIMINGS
        template<typename BufferMem>
        static void unpack(BufferMem& m)
#else
        template<typename BufferMem,typename... Args>
        static void unpack(BufferMem& m, Args&&...)
#endif
        {
            auto policy = std::launch::async;
            std::vector<std::future<void>> futures(m.m_recv_futures.size());
            std::vector<std::size_t> index_list(m.m_recv_futures.size());
            for (std::size_t i = 0; i < index_list.size(); ++i)
                index_list[i] = i;
            std::size_t size = index_list.size();
            while(size>0u)
            {
                for (std::size_t j = 0; j < size; ++j)
                {
                    const auto k = index_list[j];
                    if (m.m_recv_futures[k].test())
                    {
                        if (j < --size)
                            index_list[j--] = index_list[size];
                        futures[k] = std::async(
                            policy,
                            [k](BufferMem& m){
                                for (const auto& fb : *m.m_recv_hooks[k].second)
                                    fb.call_back(m.m_recv_hooks[k].first + fb.offset, *fb.index_container,(void*)0);
                            }, 
                            std::ref(m));
                    }
                }
            }
            for (auto& f: futures)
                f.get();
        }
    };

#ifdef __CUDACC__
    template<>
    struct packer<device::gpu>
    {
        template<typename Map, typename Futures, typename Communicator>
        static void pack(Map& map, Futures& send_futures,Communicator& comm)
        {
            std::size_t num_streams = 0;
            for (auto& p0 : map.send_memory)
            {
                for (auto& p1: p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        p1.second.buffer.resize(p1.second.size);
                        ++num_streams;
                    }
                }
            }
            std::vector<cudaStream_t> streams(num_streams);
            for (auto& x : streams) 
                cudaStreamCreate(&x);
            num_streams = 0;
            for (auto& p0 : map.send_memory)
            {
                for (auto& p1: p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        for (const auto& fb : p1.second.field_buffers)
                            fb.call_back( p1.second.buffer.data() + fb.offset, *fb.index_container, (void*)(&streams[num_streams]));
                    }
                }
            }
            num_streams=0;
            for (auto& p0 : map.send_memory)
            {
                for (auto& p1: p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        cudaStreamSynchronize(streams[num_streams]);
                        //std::cout << "isend(" << p1.second.address << ", " << p1.second.tag 
                        //<< ", " << p1.second.buffer.size() << ")" << std::endl;
                        send_futures.push_back(comm.isend(
                            p1.second.address,
                            p1.second.tag,
                            p1.second.buffer));
                        ++num_streams;
                    }
                }
            }
            for (auto& x : streams) 
                cudaStreamDestroy(x);
        }

#ifndef GHEX_COMM_2_TIMINGS
        template<typename BufferMem>
        static void unpack(BufferMem& m)
#else
        template<typename BufferMem,typename Timings>
        static void unpack(BufferMem& m, Timings& t)
#endif
        {
            auto policy = std::launch::async;
            //std::vector<std::future<void>> futures(m.m_recv_futures.size());
            std::vector<cudaStream_t> streams(m.m_recv_futures.size());
            std::vector<std::size_t> index_list(m.m_recv_futures.size());
            std::size_t i = 0;
            for (auto& x : streams)
            {
                cudaStreamCreate(&x);
                index_list[i] = i;
                ++i;
            }
            std::size_t size = index_list.size();
            while(size>0u)
            {
                for (std::size_t j = 0; j < size; ++j)
                {
                    const auto k = index_list[j];
                    if (m.m_recv_futures[k].test())
                    {
                        if (j < --size)
                            index_list[j--] = index_list[size];
                        //futures[k] = std::async(
                        //    policy,
                        //    [k](BufferMem& m, cudaStream_t& s){
                        //        for (const auto& fb : *m.m_recv_hooks[k].second)
                        //            fb.call_back(m.m_recv_hooks[k].first + fb.offset, *fb.index_container, (void*)(&s)); 
                        //    }, 
                        //    std::ref(m), std::ref(streams[k]));
#ifdef GHEX_COMM_2_TIMINGS
                        auto t0 = Timings::clock_type::now(); 
#endif
                        for (const auto& fb : *m.m_recv_hooks[k].second)
                            fb.call_back(m.m_recv_hooks[k].first + fb.offset, *fb.index_container, (void*)(&streams[k]));
#ifdef GHEX_COMM_2_TIMINGS
                        auto t1 = Timings::clock_type::now(); 
                        t.unpack_launch_time += t1-t0;
#endif
                    }
                }
            }
            //int k = 0;
            for (auto& x : streams) 
            {
                //futures[k++].get();
                cudaStreamSynchronize(x);
                cudaStreamDestroy(x);
            }
        }
    };
#endif
 
    /** @brief communication object responsible for exchanging halo data. Allocates storage depending on the 
     * device type and device id of involved fields.
     * @tparam P message protocol type
     * @tparam GridType grid tag type
     * @tparam DomainIdType domain id type*/
    template<typename P, typename GridType, typename DomainIdType>
    class communication_object
    {
    private: // friend class
        friend class communication_handle<P,GridType,DomainIdType>;

    public: // member types
        /** @brief handle type returned by exhange operation */
        using handle_type             = communication_handle<P,GridType,DomainIdType>;
        using domain_id_type          = DomainIdType;

    private: // member types
        using communicator_type       = typename handle_type::communicator_type;
        using address_type            = typename communicator_type::address_type;

        using pattern_type            = pattern<P,GridType,DomainIdType>;
        using index_container_type    = typename pattern_type::index_container_type;
        using pack_function_type      = std::function<void(void*,const index_container_type&, void*)>;
        using unpack_function_type    = std::function<void(const void*,const index_container_type&, void*)>;

        template<typename D, typename F>
        using buffer_info_type        = buffer_info<pattern_type,D,F>;

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

        template<typename Function>
        struct field_buffer
        {
            Function call_back;
            const typename pattern_type::map_type::mapped_type* index_container;
            std::size_t offset;
        };

        // holds actual memory
        template<class Vector, class Function>
        struct buffer
        {
            using field_buffer_type = field_buffer<Function>;
            address_type address;
            int tag;
            Vector buffer;
            std::size_t size;
            std::vector<field_buffer_type> field_buffers;
        };

        // one instance will be created per device type, memory is organized in a map
        template<typename Device>
        struct buffer_memory
        {
            using device_type = Device;
            using id_type     = typename device_type::id_type;
            using vector_type = typename device_type::template vector_type<char>;
            
            using send_buffer_type = buffer<vector_type,pack_function_type>; 
            using recv_buffer_type = buffer<vector_type,unpack_function_type>; 
            using send_memory_type = std::map<id_type, std::map<domain_id_pair,send_buffer_type>>;
            using recv_memory_type = std::map<id_type, std::map<domain_id_pair,recv_buffer_type>>;

            send_memory_type send_memory;
            recv_memory_type recv_memory;

            std::vector<typename communicator_type::template future<void>> m_recv_futures;
            std::vector<std::pair<char*,std::vector<field_buffer<unpack_function_type>>*>> m_recv_hooks;
            std::vector<bool> m_completed_hooks;
        };
        
        // tuple type of buffer_memory (one element for each device in device::device_list)
        using memory_type = detail::transform<device::device_list>::with<buffer_memory>;

    private: // members
        bool m_valid;
        std::map<const typename pattern_type::pattern_container_type*, int> m_max_tag_map;
        memory_type m_mem;
        std::vector<typename communicator_type::template future<void>> m_send_futures;

    public: // ctors
        /** @brief construct a communication object from a message tag map
         * @param max_tag_map a map which holds the maximum global message tag for each distinct pattern_container 
         * instance */
        communication_object(const std::map<const typename pattern_type::pattern_container_type*, int>& max_tag_map)
        : m_valid(false), m_max_tag_map(max_tag_map) {}

        communication_object(const communication_object&) = delete;
        communication_object(communication_object&&) = default;

    public: // member functions
        /**
         * @brief blocking variant of halo exchange
         * @tparam Devices list of device types
         * @tparam Fields list of field types
         * @param buffer_infos buffer_info objects created by binding a field descriptor to a pattern
         */
        template<typename... Devices, typename... Fields>
        void bexchange(buffer_info_type<Devices,Fields>... buffer_infos)
        {
            exchange(buffer_infos...).wait();
        }

        /**
         * @brief non-blocking exchange of halo data
         * @tparam Devices list of device types
         * @tparam Fields list of field types
         * @param buffer_infos buffer_info objects created by binding a field descriptor to a pattern
         * @return handle to await communication
         */
        template<typename... Devices, typename... Fields>
        [[nodiscard]] handle_type exchange(buffer_info_type<Devices,Fields>... buffer_infos)
        {
            if (m_valid) throw std::runtime_error("earlier exchange operation was not finished");
            m_valid = true;

            using buffer_infos_ptr_t     = std::tuple<std::remove_reference_t<decltype(buffer_infos)>*...>;
            using memory_t               = std::tuple<buffer_memory<Devices>*...>;

            buffer_infos_ptr_t buffer_info_tuple{&buffer_infos...};
            handle_type h(*this,std::get<0>(buffer_info_tuple)->get_pattern().communicator());
            memory_t memory_tuple{&(std::get<buffer_memory<Devices>>(m_mem))...};
            int tag_offsets[sizeof...(Fields)] = { m_max_tag_map[&(buffer_infos.get_pattern_container())]... };

            int i = 0;
            detail::for_each(memory_tuple, buffer_info_tuple, [this,&i,&tag_offsets](auto mem, auto bi) 
            {
                using device_type = typename std::remove_reference_t<decltype(*mem)>::device_type;
                using value_type  = typename std::remove_reference_t<decltype(*bi)>::value_type;

                auto field_ptr = &(bi->get_field());
                const domain_id_type my_dom_id = bi->get_field().domain_id();

                this->allocate<device_type,value_type,typename buffer_memory<device_type>::recv_buffer_type>(
                    mem->recv_memory[bi->device_id()],
                    bi->get_pattern().recv_halos(),
                    [field_ptr](const void* buffer, const index_container_type& c, void* arg) 
                        { field_ptr->unpack(reinterpret_cast<const value_type*>(buffer),c,arg); },
                    my_dom_id,
                    bi->device_id(),
                    tag_offsets[i],
                    true);
                this->allocate<device_type,value_type,typename buffer_memory<device_type>::send_buffer_type>(
                    mem->send_memory[bi->device_id()],
                    bi->get_pattern().send_halos(),
                    [field_ptr](void* buffer, const index_container_type& c, void* arg) 
                        { field_ptr->pack(reinterpret_cast<value_type*>(buffer),c,arg); },
                    my_dom_id,
                    bi->device_id(),
                    tag_offsets[i],
                    false);
                ++i;
            });
            post(h.m_comm);
            return h; 
        }

    private:
        void post(communicator_type& comm)
        {
            detail::for_each(m_mem, [this,&comm](auto& m)
            {
                for (auto& p0 : m.recv_memory)
                {
                    for (auto& p1: p0.second)
                    {
                        if (p1.second.size > 0u)
                        {
                            p1.second.buffer.resize(p1.second.size);
                            //std::cout << "rank " << comm.rank() << ": irecv(" << p1.second.address << ", " << p1.second.tag 
                            //<< ", " << p1.second.buffer.size() << ")" << std::endl;
                            m.m_recv_futures.push_back(comm.irecv(
                                p1.second.address,
                                p1.second.tag,
                                p1.second.buffer.data(),
                                p1.second.buffer.size()));
                            m.m_recv_hooks.push_back(std::make_pair(p1.second.buffer.data(),&(p1.second.field_buffers))); 
                            m.m_completed_hooks.push_back(false);
                        }
                    }
                }
            });
            detail::for_each(m_mem, [this,&comm](auto& m)
            {
                using device_type = typename std::remove_reference_t<decltype(m)>::device_type;
                packer<device_type>::pack(m,m_send_futures,comm);
            });
        }

        void wait()
        {
            if (!m_valid) return;

            detail::for_each(m_mem, [this](auto& m)
            {
                using device_type = typename std::remove_reference_t<decltype(m)>::device_type;
                packer<device_type>::unpack(m);
            });

            for (auto& f : m_send_futures) 
                f.wait();

            clear();
        }
#ifdef GHEX_COMM_2_TIMINGS
        template<typename Timings>
        void wait(Timings& t)
        {
            if (!m_valid) return;

            detail::for_each(m_mem, [this, t](auto& m)
            {
                using device_type = typename std::remove_reference_t<decltype(m)>::device_type;
                packer<device_type>::unpack(m, t);
            });

            for (auto& f : m_send_futures) 
                f.wait();

            clear();
        }
#endif

        // clear the internal flags so that a new exchange can be started
        // important: does not deallocate
        void clear()
        {
            m_valid = false;
            m_send_futures.clear();
            detail::for_each(m_mem, [this](auto& m)
            {
                m.m_recv_futures.clear();
                m.m_recv_hooks.resize(0);
                m.m_completed_hooks.resize(0);
                for (auto& p0 : m.send_memory)
                    for (auto& p1 : p0.second)
                    {
                        p1.second.buffer.resize(0);
                        p1.second.size = 0;
                        p1.second.field_buffers.resize(0);
                    }
                for (auto& p0 : m.recv_memory)
                    for (auto& p1 : p0.second)
                    {
                        p1.second.buffer.resize(0);
                        p1.second.size = 0;
                        p1.second.field_buffers.resize(0);
                    }
            });
        }

        // compute memory requirements to be allocated on the device
        template<typename Device, typename ValueType, typename BufferType, typename Memory, typename Halos, typename Function, typename DeviceIdType>
        void allocate(Memory& memory, const Halos& halos, Function&& func, domain_id_type my_dom_id, DeviceIdType device_id, 
                      int tag_offset, bool receive)
        {
            for (const auto& p_id_c : halos)
            {
                const auto num_elements   = pattern_type::num_elements(p_id_c.second);
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
                            Device::template make_vector<char>(device_id),
                            0,
                            std::vector<typename BufferType::field_buffer_type>()
                        })).first;
                }
                else if (it->second.size==0)
                {
                    it->second.address = remote_address;
                    it->second.tag = p_id_c.first.tag+tag_offset;
                    it->second.field_buffers.resize(0);
                }
                const auto prev_size = it->second.size;
                const auto padding = ((prev_size+alignof(ValueType)-1)/alignof(ValueType))*alignof(ValueType) - prev_size;
                it->second.field_buffers.push_back(
                    typename BufferType::field_buffer_type{std::forward<Function>(func), &p_id_c.second, prev_size + padding});
                it->second.size += padding + static_cast<std::size_t>(num_elements)*sizeof(ValueType);
            }
        }
    };

    namespace detail {

        // helper template metafunction to test equality of a type with respect to all element types of a tuple
        template<typename Test, typename... Ts>
        struct test_eq_t {};

        template<typename Test, typename T0, typename T1, typename... Ts>
        struct test_eq_t<Test,T0,T1,Ts...> : public 
            std::integral_constant<
                bool, 
                std::is_same<Test,T0>::value && test_eq_t<Test,T1,Ts...>::value
            > {};

        template<typename Test, typename T0>
        struct test_eq_t<Test,T0> : public 
            std::integral_constant<bool, std::is_same<Test,T0>::value> {};

    } // namespace detail

    /** @brief creates a communication object based on the patterns involved
     * @tparam Patterns list of pattern types
     * @param ... unnamed list of pattern_holder objects
     * @return communication object */
    template<typename... Patterns>
    auto make_communication_object(const Patterns&... patterns)
    {
        using ps_t = std::tuple<typename Patterns::value_type...>;
        using p_t  = std::tuple_element_t<0,ps_t>;
        using protocol_type    = typename p_t::communicator_type::protocol_type;
        using grid_type        = typename p_t::grid_type;
        using domain_id_type   = typename p_t::domain_id_type;

        using test_t = pattern_container<protocol_type,grid_type,domain_id_type>;
        static_assert(detail::test_eq_t<test_t,Patterns...>::value, "patterns are incompatible");

        // test for repeating patterns by looking at the patterns address
        // if repetitions are found, the tag offset is not increased
        const test_t* ptrs[sizeof...(Patterns)] = { &patterns... };
        std::map<const test_t*,int> pat_ptr_map;
        int max_tag = 0;
        for (unsigned int k=0; k<sizeof...(Patterns); ++k)
        {
            auto p_it_bool = pat_ptr_map.insert( std::make_pair(ptrs[k], max_tag) );
            if (p_it_bool.second == true)
                max_tag += ptrs[k]->max_tag()+1;
        }

        return communication_object<protocol_type,grid_type,domain_id_type>(pat_ptr_map);
    }
        
} // namespace gridtools

#endif /* INCLUDED_COMMUNICATION_OBJECT_2_HPP */

