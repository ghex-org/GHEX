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
#ifndef INCLUDED_COMMUNICATION_OBJECT_ERASED_HPP
#define INCLUDED_COMMUNICATION_OBJECT_ERASED_HPP


#ifdef GRIDTOOLS_COMM_OBJECT_THREAD_SAFE
    #define GRIDTOOLS_TAG_OFFSET 100
#else
    #define GRIDTOOLS_TAG_OFFSET 100
#endif

#include "utils.hpp"
#include "buffer_info.hpp"
#include "devices.hpp"

namespace gridtools {

    namespace detail {

        template<typename Tuple>
        struct transform {};

        template<template<typename...> typename Tuple, typename... Ts>
        struct transform<Tuple<Ts...>>
        {
            template<template<typename> typename CT>
            using with = Tuple<CT<Ts>...>;
        };

    } // namespace detail

    // forward declaration
    template<typename P, typename GridType, typename DomainIdType>
    class communication_object_erased;

    /** @brief handle type for waiting on asynchronous communication processes
     * @tparam P message protocol type
     * @tparam GridType grid tag type
     * @tparam DomainIdType domain id type*/
    template<typename P, typename GridType, typename DomainIdType>
    class communication_handle
    {
    private: // friend class
        friend class communication_object_erased<P,GridType,DomainIdType>;

    private: // member types
        using co_t                    = communication_object_erased<P,GridType,DomainIdType>;
        using communicator_type       = protocol::communicator<P>;
        using address_type            = typename communicator_type::address_type;
        using pattern_type            = pattern<P,GridType,DomainIdType>;
        using index_container_type    = typename pattern_type::index_container_type;
        using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
        using pack_function_type      = std::function<void(void*,const index_container_type&)>;
        using unpack_function_type    = std::function<void(const void*,const index_container_type&)>;

        // holds bookkeeping info for a data field
        // - type erased function pointers to pack and unpack methods
        // - pointers to memory locations
        struct field
        {
            struct chunk_type
            {
                std::size_t m_offset;
                void* m_buffer;
            };
            using chunk_vector_type = std::vector<chunk_type>;
            const pattern_type* m_pattern;
            pack_function_type m_pack;
            unpack_function_type m_unpack;
            chunk_vector_type m_recv_chunks;
            chunk_vector_type m_send_chunks;
        };

        /*struct field_send_chunk
        {
            std::size_t m_offset;
            pack_function_type m_fct;
            const typename pattern_type::map_type::mapped_type* m_is_vec;
        };
        struct field_recv_chunk
        {
            std::size_t m_offset;
            unpack_function_type m_fct;
            const typename pattern_type::map_type::mapped_type* m_is_vec;
        };*/

    private: // private constructor
        communication_handle(co_t& co, const communicator_type& comm, std::size_t size) : m_co{&co}, m_comm{comm}, m_fields(size) {}

    public: // copy and move ctors
        communication_handle(communication_handle&&) = default;
        communication_handle(const communication_handle&) = delete;

    public: // member function
        /** @brief  wait for communication to be finished*/
        void wait()
        {
            for (auto& f : m_futures) f.wait();
            unpack();
            m_co->clear();
        }

    private: // member functions
        void post()
        {
            pack();
            /*detail::for_each(m_co->m_mem, [this](auto& m)
            {
                for (auto& mm : m.recv_memory)
                    for (auto& p : mm.second)
                        if (p.second.size()>0)
                        {
                            //std::cout << "irecv(" << p.first.address << ", " << p.first.tag << ", " << p.second.size() << ")" << std::endl;
                            //m_futures.push_back(m_comm.irecv(p.first.address, p.first.tag, p.second.data(), p.second.size()));
                        }
                for (auto& mm : m.send_memory)
                    for (auto& p : mm.second)
                        if (p.second.size()>0)
                        {
                            //std::cout << "isend(" << p.first.address << ", " << p.first.tag << ", " << p.second.size() << ")" << std::endl;
                            //m_futures.push_back(m_comm.isend(p.first.address, p.first.tag, p.second));
                        }
            });*/
            //using memory_type2 = std::map<id_type, std::map<address_type, std::map<domain_id_pair, vector_type>>>;
            detail::for_each(m_co->m_mem, [this](auto& m)
            {
                for (auto& p0 : m.recv_memory3)
                    // pair<id_type, std::map<...
                    for (auto& p1 : p0.second)
                    {
                        // pair<domain_id_pair, pair<remote_id, vector_type>>
                        if (p1.second.second.size()>0)
                        {
                            std::cout << "irecv2(" << p1.second.first.address << ", " << p1.second.first.tag << ", " << p1.second.second.size() << ")" << std::endl;
                            m_futures.push_back(m_comm.irecv(p1.second.first.address, p1.second.first.tag, p1.second.second.data(), p1.second.second.size()));
                        }
                        /*int tag = 1000;
                        for (auto& p2 : p1.second)
                        {
                            // pair<domain_id_par, vector_type
                            if (p2.second.size()>0)
                            {
                                std::cout << "irecv2(" << p1.first << ", " << tag << ", " << p2.second.size() << ")" << std::endl;
                                m_futures.push_back(m_comm.irecv(p1.first, tag, p2.second.data(), p2.second.size()));
                                ++tag;
                            }
                        }*/
                    }
                for (auto& p0 : m.send_memory3)
                    // pair<id_type, std::map<...
                    for (auto& p1 : p0.second)
                    {
                        // pair<address_type, std::map
                        if (p1.second.second.size()>0)
                        {
                            std::cout << "isend2(" << p1.second.first.address << ", " << p1.second.first.tag << ", " << p1.second.second.size() << ")" << std::endl;
                            m_futures.push_back(m_comm.isend(p1.second.first.address, p1.second.first.tag, p1.second.second));
                        }
                        /*int tag = 1000;
                        for (auto& p2 : p1.second)
                        {
                            // pair<domain_id_par, vector_type
                            if (p2.second.size()>0)
                            {
                                std::cout << "isend2(" << p1.first << ", " << tag << ", " << p2.second.size() << ")" << std::endl;
                                m_futures.push_back(m_comm.isend(p1.first, tag, p2.second));
                                ++tag;
                            }
                        }*/
                    }
            });
        }

        void pack()
        {
            // should I try to use inverse map to pack buffer by buffer?
            for (auto& f : m_fields)
            {
                std::size_t k=0;
                for (const auto& p_id_c : f.m_pattern->send_halos())
                {
                    //std::cout << f.m_send_chunks.size() << std::endl;
                    f.m_pack(f.m_send_chunks[k++].m_buffer, p_id_c.second);
                }
            }
        }

        void unpack()
        {
            for (auto& f : m_fields)
            {
                std::size_t k=0;
                for (const auto& p_id_c : f.m_pattern->recv_halos())
                    f.m_unpack(f.m_recv_chunks[k++].m_buffer, p_id_c.second);
            }
        }

    private: // members
        co_t* m_co;
        communicator_type m_comm;
        std::vector<field> m_fields;
        std::vector<typename communicator_type::template future<void>> m_futures;
    };

    /** @brief communication object responsible for exchanging halo data. Allocates storage depending on the 
     * device type and device id of involved fields.
     * @tparam P message protocol type
     * @tparam GridType grid tag type
     * @tparam DomainIdType domain id type*/
    template<typename P, typename GridType, typename DomainIdType>
    class communication_object_erased
    {
    private: // friend class
        friend class communication_handle<P,GridType,DomainIdType>;

    public: // member types
        /** @brief handle type returned by exhange operation */
        using handle_type             = communication_handle<P,GridType,DomainIdType>;

    private: // member types
        using communicator_type       = typename handle_type::communicator_type;
        using address_type            = typename communicator_type::address_type;
        using pattern_type            = typename handle_type::pattern_type;
        using index_container_type    = typename handle_type::index_container_type;
        using extended_domain_id_type = typename handle_type::extended_domain_id_type;
        using domain_id_type          = DomainIdType;
        //using field_send_chunk        = typename handle_type::field_send_chunk;
        //using field_recv_chunk        = typename handle_type::field_recv_chunk;

        template<typename D, typename F>
        using buffer_info_type = buffer_info<pattern_type,D,F>;

        struct domain_id_pair
        {
            domain_id_type my_id;
            domain_id_type remote_id;
            bool operator<(const domain_id_pair& other) const noexcept
            {
                return (my_id < other.my_id ? true : (my_id > other.my_id ? false : (remote_id<other.remote_id)));
            }
        };

        struct remote_id
        {
            address_type address;
            int tag;
        };

        // holds actual memory
        // one instance will be created per device type
        // memory is organized in a map:
        // map( device_id -> 
        //                   map( extended_domain_id ->
        //                                               vector of chars (device specific) ))
        template<typename Device>
        struct buffer_memory
        {
            using device_type = Device;
            using id_type = typename device_type::id_type;
            using vector_type = typename device_type::template vector_type<char>;
            //struct extended_domain_id_comp
            //{
            //    bool operator()(const extended_domain_id_type& l, const extended_domain_id_type& r) const noexcept
            //    {
            //        //return (l.id < r.id ? true : (l.id == r.id ? (l.tag < r.tag) : false));
            //        return (l.mpi_rank < r.mpi_rank ? true : (l.mpi_rank == r.mpi_rank ? (l.tag < r.tag) : false));
            //        /*return 
            //            (l.mpi_rank < r.mpi_rank ? 
            //                true 
            //            : 
            //                (l.mpi_rank == r.mpi_rank ? 
            //                    (l.id < r.id ?
            //                        true        
            //                    :
            //                        (l.id == r.id ?
            //                            (l.tag < r.tag)
            //                        : // l.mpi_rank == r.mpi_rank && l.id > r.id
            //                            false
            //                        )
            //                    )
            //                : // l.mpi_rank > r.mpi_rank
            //                    false
            //                )
            //            );*/
            //    }
            //};
            //using memory_type = std::map<
            //    id_type,
            //    std::map<
            //        extended_domain_id_type,
            //        vector_type,
            //        extended_domain_id_comp
            //    >
            //>;
            //memory_type recv_memory;
            //memory_type send_memory;

            //using memory_type2 = std::map<id_type, std::map<address_type, std::map<domain_id_pair, vector_type>>>;
            //memory_type2 recv_memory2;
            //memory_type2 send_memory2;

            using memory_type3 = std::map<id_type, std::map<domain_id_pair, std::pair<remote_id,vector_type>>>;
            memory_type3 recv_memory3;
            memory_type3 send_memory3;

            /*using send_memory_type = std::map<
                id_type,
                std::map<
                    extended_domain_id_type,
                    std::pair<vector_type, std::vector<field_send_chunk>>,
                    extended_domain_id_comp
                >
            >;
            using recv_memory_type = std::map<
                id_type,
                std::map<
                    extended_domain_id_type,
                    std::pair<vector_type, std::vector<field_recv_chunk>>,
                    extended_domain_id_comp
                >
            >;
            send_memory_type send_memory3;
            recv_memory_type recv_memory3;*/
        };

        // tuple type of buffer_memory (one element for each device in device::device_list)
        using memory_type = detail::transform<device::device_list>::with<buffer_memory>;

    private: // members
        std::map<const typename pattern_type::pattern_container_type*, int> m_max_tag_map;
        memory_type m_mem;

    public: // ctors

        communication_object_erased(const std::map<const typename pattern_type::pattern_container_type*, int>& max_tag_map)
        : m_max_tag_map(max_tag_map) {}
        communication_object_erased(const communication_object_erased&) = delete;
        communication_object_erased(communication_object_erased&&) = default;
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
            using buffer_infos_ptr_t  = std::tuple<std::remove_reference_t<decltype(buffer_infos)>*...>;
            using memory_t   = std::tuple<buffer_memory<Devices>*...>;

            buffer_infos_ptr_t buffer_info_tuple{&buffer_infos...};
            handle_type h(*this,std::get<0>(buffer_info_tuple)->get_pattern().communicator(), sizeof...(Fields));
            memory_t memory_tuple{&(std::get<buffer_memory<Devices>>(m_mem))...};

            using pattern_container_type = typename pattern_type::pattern_container_type;

            /*const pattern_container_type* pat_ptr[sizeof...(Fields)] = { &(buffer_infos.get_pattern_container())... };
            int tag_offsets[sizeof...(Fields)];
            int max_tag_offset = 0;
            std::map<const pattern_container_type*,int> pat_ptr_map;
            for (unsigned int k=0; k<sizeof...(Fields); ++k)
            {
                auto p_it_bool = pat_ptr_map.insert( std::make_pair( pat_ptr[k], max_tag_offset) );
                if (p_it_bool.second == true)
                {
                    //std::cout << "new pattern! " << pat_ptr[k] << std::endl;
                    // insertion took place
                    tag_offsets[k] = max_tag_offset;
                    max_tag_offset += GRIDTOOLS_TAG_OFFSET;
                }
                else
                {
                    //std::cout << "existing pattern! " << pat_ptr[k] << std::endl;
                    // element already present
                    tag_offsets[k] = p_it_bool.first->second;
                }
                //std::cout << "tag_offset[" << k << "] = " << tag_offsets[k] << std::endl;
            }*/

            int tag_offsets[sizeof...(Fields)] = { m_max_tag_map[&(buffer_infos.get_pattern_container())]... };

            int i = 0;
            detail::for_each(memory_tuple, buffer_info_tuple, [this,&i,&h,&tag_offsets](auto mem, auto bi) 
            {
                using device_type = typename std::remove_reference_t<decltype(*mem)>::device_type;
                using value_type  = typename std::remove_reference_t<decltype(*bi)>::value_type;
                auto& f     = h.m_fields[i];
                f.m_pattern = &(bi->get_pattern());
                auto field_ptr = &(bi->get_field());
                f.m_pack    = [field_ptr](void* buffer, const index_container_type& c) 
                                  { field_ptr->pack(reinterpret_cast<value_type*>(buffer),c); };
                f.m_unpack  = [field_ptr](const void* buffer, const index_container_type& c) 
                                  { field_ptr->unpack(reinterpret_cast<const value_type*>(buffer),c); };
                f.m_recv_chunks.resize(f.m_pattern->recv_halos().size());
                f.m_send_chunks.resize(f.m_pattern->send_halos().size());
                //auto& m_recv = mem->recv_memory[bi->device_id()];
                //auto& m_send = mem->send_memory[bi->device_id()];
                
                //std::cout << "allocating recv memory" << std::endl;
                //allocate<device_type, value_type>(f.m_pattern->recv_halos(), m_recv, bi->device_id(), f.m_recv_chunks,i*GRIDTOOLS_TAG_OFFSET);
                //allocate<device_type, value_type>(f.m_pattern->recv_halos(), m_recv, bi->device_id(), f.m_recv_chunks,tag_offsets[i]);

                //std::cout << "allocating send memory" << std::endl;
                //allocate<device_type, value_type>(f.m_pattern->send_halos(), m_send, bi->device_id(), f.m_send_chunks,i*GRIDTOOLS_TAG_OFFSET);
                //allocate<device_type, value_type>(f.m_pattern->send_halos(), m_send, bi->device_id(), f.m_send_chunks,tag_offsets[i]); 

                const domain_id_type my_dom_id = bi->get_field().domain_id();
                std::size_t j=0;
                auto& m_recv3 = mem->recv_memory3[bi->device_id()];
                for (const auto& p_id_c : bi->get_pattern().recv_halos())
                {
                    const auto remote_address = p_id_c.first.address;
                    const auto remote_domain_id = p_id_c.first.id;
                    const auto d_p = domain_id_pair{my_dom_id,remote_domain_id};
                    auto it = m_recv3.find(d_p);
                    if (it == m_recv3.end())
                    {
                        std::cout << "new recv pair found: " << my_dom_id << " " << remote_domain_id << std::endl;
                        it = m_recv3.insert(std::make_pair(d_p, std::make_pair(remote_id{remote_address, p_id_c.first.tag+tag_offsets[i]}, device_type::template make_vector<char>(bi->device_id())))).first;
                    }
                    else if (it->second.second.size()==0)
                    {
                        it->second.first.address = remote_address;
                        it->second.first.tag = p_id_c.first.tag+tag_offsets[i];
                    }
                    const auto prev_size = it->second.second.size();
                    //chunks[j].m_offset = prev_size;
                    f.m_recv_chunks[j].m_offset = prev_size;
                    it->second.second.resize(0);
                    it->second.second.resize(prev_size + alignof(value_type) +
                        static_cast<std::size_t>(pattern_type::num_elements(p_id_c.second))*sizeof(value_type));
                    ++j;
                }
                j=0;
                auto& m_send3 = mem->send_memory3[bi->device_id()];
                for (const auto& p_id_c : bi->get_pattern().send_halos())
                {
                    const auto remote_address = p_id_c.first.address;
                    const auto remote_domain_id = p_id_c.first.id;
                    const auto d_p = domain_id_pair{remote_domain_id, my_dom_id};
                    auto it = m_send3.find(d_p);
                    if (it == m_send3.end())
                    {
                        //std::cout << "new send pair found: " << remote_domain_id << " " << my_dom_id<< std::endl;
                        //it = m1.insert(std::make_pair(d_p, device_type::template make_vector<char>(bi->device_id()))).first;
                        it = m_send3.insert(std::make_pair(d_p, std::make_pair(remote_id{remote_address, p_id_c.first.tag+tag_offsets[i]}, device_type::template make_vector<char>(bi->device_id())))).first;
                    }
                    else if (it->second.second.size()==0)
                    {
                        it->second.first.address = remote_address;
                        it->second.first.tag = p_id_c.first.tag+tag_offsets[i];
                    }
                    const auto prev_size = it->second.second.size();
                    //chunks[j].m_offset = prev_size;
                    f.m_send_chunks[j].m_offset = prev_size;
                    it->second.second.resize(0);
                    it->second.second.resize(prev_size + alignof(value_type) +
                        static_cast<std::size_t>(pattern_type::num_elements(p_id_c.second))*sizeof(value_type));
                    ++j;
                }
                ++i;
            });

            i = 0;
            detail::for_each(memory_tuple, buffer_info_tuple, [this,&i,&h,&tag_offsets](auto mem, auto bi) 
            {
                using device_type = typename std::remove_reference_t<decltype(*mem)>::device_type;
                using value_type  = typename std::remove_reference_t<decltype(*bi)>::value_type;
                auto& f     = h.m_fields[i];
                //auto& m_recv = mem->recv_memory[bi->device_id()];
                //auto& m_send = mem->send_memory[bi->device_id()];

                //align<device_type, value_type>(f.m_pattern->recv_halos(), m_recv, bi->device_id(), f.m_recv_chunks,i*GRIDTOOLS_TAG_OFFSET);
                //align<device_type, value_type>(f.m_pattern->recv_halos(), m_recv, bi->device_id(), f.m_recv_chunks,tag_offsets[i]);

                //align<device_type, value_type>(f.m_pattern->send_halos(), m_send, bi->device_id(), f.m_send_chunks,i*GRIDTOOLS_TAG_OFFSET);
                //align<device_type, value_type>(f.m_pattern->send_halos(), m_send, bi->device_id(), f.m_send_chunks,tag_offsets[i]);


                const domain_id_type my_dom_id = bi->get_field().domain_id();
                std::size_t j=0;
                auto& m_recv3 = mem->recv_memory3[bi->device_id()];
                for (const auto& p_id_c : bi->get_pattern().recv_halos())
                {
                    const auto remote_address = p_id_c.first.address;
                    const auto remote_domain_id = p_id_c.first.id;
                    const auto d_p = domain_id_pair{my_dom_id,remote_domain_id};
                    auto it = m_recv3.find(d_p);
                    auto& vec = it->second.second;
                    //chunks[j].m_buffer = Device::template align<value_type>(vec.data()+chunks[j].m_offset, device_id);
                    f.m_recv_chunks[j].m_buffer = device_type::template align<value_type>(
                        vec.data() + f.m_recv_chunks[j].m_offset, bi->device_id());
                    ++j;
                }
                j=0;
                auto& m_send3 = mem->send_memory3[bi->device_id()];
                for (const auto& p_id_c : bi->get_pattern().recv_halos())
                {
                    const auto remote_address = p_id_c.first.address;
                    const auto remote_domain_id = p_id_c.first.id;
                    const auto d_p = domain_id_pair{remote_domain_id, my_dom_id};
                    auto it = m_send3.find(d_p);
                    auto& vec = it->second.second;
                    //chunks[j].m_buffer = Device::template align<value_type>(vec.data()+chunks[j].m_offset, device_id);
                    f.m_send_chunks[j].m_buffer = device_type::template align<value_type>(
                        vec.data() + f.m_send_chunks[j].m_offset, bi->device_id());
                    ++j;
                }


                ++i;
            });

            h.post();
            return std::move(h);
        }

    private: // member functions
        // allocate memory on device
        template<typename Device, typename ValueType>
        void allocate(
            const typename pattern_type::map_type& halos, 
            typename buffer_memory<Device>::memory_type::mapped_type& memory, 
            typename Device::id_type device_id, 
            typename handle_type::field::chunk_vector_type& chunks, int tag_offset)
        {
            std::size_t j=0;
            for (const auto& p_id_c : halos)
            {
                auto key = p_id_c.first;

                /*auto it_f = std::find(memory.begin(), memory.end(), 
                    [](const auto& p)
                    {
                        p.first
                    });*/

                key.tag += tag_offset;
                //auto it = memory.find(p_id_c.first);
                auto it = memory.find(key);
                if (it == memory.end())
                {
                    //std::cout << "allocating memory for key " << p_id_c.first << std::endl;
                    //std::cout << "allocating memory for key " << key << std::endl;
                    //it = memory.insert(std::make_pair(p_id_c.first, Device::template make_vector<char>(device_id))).first;
                    it = memory.insert(std::make_pair(key, Device::template make_vector<char>(device_id))).first;
                }
                const auto prev_size = it->second.size();
                chunks[j].m_offset = prev_size;
                it->second.resize(0);
                it->second.resize(prev_size + alignof(ValueType) +
                                  static_cast<std::size_t>(pattern_type::num_elements(p_id_c.second))*sizeof(ValueType));
                ++j;
            }
        }

        // align memory on device
        template<typename Device, typename ValueType>
        void align(
            const typename pattern_type::map_type& halos, 
            typename buffer_memory<Device>::memory_type::mapped_type& memory, 
            typename Device::id_type device_id, 
            typename handle_type::field::chunk_vector_type& chunks, int tag_offset)
        {
            int j=0;
            for (const auto& p_id_c : halos)
            {
                auto key = p_id_c.first;
                key.tag += tag_offset;
                //auto& vec = memory[p_id_c.first];
                auto& vec = memory[key];
                chunks[j].m_buffer = Device::template align<ValueType>(vec.data()+chunks[j].m_offset, device_id);
                ++j;
            }
        }

        // reset storage (but doesn't deallocate)
        void clear()
        {
            detail::for_each(m_mem, [](auto& m)
            {
                /*for (auto& p0 : m.recv_memory)
                    for (auto& p1 : p0.second)
                        p1.second.resize(0);
                for (auto& p0 : m.send_memory)
                    for (auto& p1 : p0.second)
                        p1.second.resize(0);*/

                for (auto& p0 : m.recv_memory3)
                    for (auto& p1 : p0.second)
                    {
                        p1.second.second.resize(0);
                    }
                for (auto& p0 : m.send_memory3)
                    for (auto& p1 : p0.second)
                    {
                        p1.second.second.resize(0);
                    }
            });
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

    /**
     * @brief creates a communication object based on the patterns involved
     * @tparam Patterns list of pattern types
     * @param ... unnamed list of pattern_holder objects
     * @return communication object
     */
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

        int max_tags[sizeof...(Patterns)] = { patterns.max_tag()... };
        const test_t* ptrs[sizeof...(Patterns)] = { &patterns... };
        std::map<const test_t*,int> pat_ptr_map;
        int max_tag = 0;
        for (unsigned int k=0; k<sizeof...(Patterns); ++k)
        {
            auto p_it_bool = pat_ptr_map.insert( std::make_pair(ptrs[k], max_tag) );
            if (p_it_bool.second == true)
            {
                // insertion took place
                max_tag += ptrs[k]->max_tag()+1+100;
                //tag_offsets[k] = max_tag_offset;
                //max_tag_offset += GRIDTOOLS_TAG_OFFSET;
            }
            else
            {
                //std::cout << "existing pattern! " << pat_ptr[k] << std::endl;
                // element already present
                //tag_offsets[k] = p_it_bool.first->second;
            }
            
        }

        return std::move(communication_object_erased<protocol_type,grid_type,domain_id_type>(pat_ptr_map));
    }

} // namespace gridtools

#endif /* INCLUDED_COMMUNICATION_OBJECT_ERASED_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

