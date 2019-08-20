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
#include "./simple_field_wrapper.hpp"
#include <thread>
#include <future>
#include <map>
#include <stdio.h>

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

    public: // public constructor
        communication_handle(const communicator_type& comm) 
        : m_comm{comm} {}

    private: // private constructor
        /*communication_handle(co_t& co, const communicator_type& comm) 
        : m_co{&co}, m_comm{comm} {}*/

        //using wait_fct_t = void (co_t::*)();
        template<typename Func>
        communication_handle(co_t& co, const communicator_type& comm, Func&& wait_fct) 
        : m_co{&co}, m_comm{comm}, m_wait_fct(std::forward<Func>(wait_fct)) {}

    public: // copy and move ctors
        communication_handle(communication_handle&&) = default;
        communication_handle(const communication_handle&) = delete;

        communication_handle& operator=(communication_handle&&) = default;
        communication_handle& operator=(const communication_handle&) = delete;

    public: // member functions
        /** @brief  wait for communication to be finished*/
        void wait() { /*if (m_co) m_co->wait();*/ if (m_wait_fct) m_wait_fct(); }
#ifdef GHEX_COMM_2_TIMINGS
        template<typename Timings>
        void wait(Timings& t) { if (m_co) m_co->wait(t); }
#endif

    private: // members
        co_t* m_co = nullptr;
        communicator_type m_comm;
        //wait_fct_t m_wait_fct = nullptr; 
        //std::function<void(co_t&, void)> m_wait_fct;
        std::function<void()> m_wait_fct;
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
            //auto policy = std::launch::async;
            //std::vector<std::future<void>> futures(m.m_recv_futures.size());
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
                        //futures[k] = std::async(
                        //    policy,
                        //    [k](BufferMem& m){
                                for (const auto& fb : *m.m_recv_hooks[k].second)
                                    fb.call_back(m.m_recv_hooks[k].first + fb.offset, *fb.index_container,(void*)0);
                        //    }, 
                        //    std::ref(m));
                    }
                }
            }
            //for (auto& f: futures)
            //    f.get();
        }
    };
	
#ifdef __CUDACC__
    
    template<class T, unsigned int N>
    struct pack_arg_t
    {
        T m_data[N];

        __host__ pack_arg_t& fill(const T* data, unsigned int size)
        {
            if (size > N) throw std::runtime_error("static space too small " + std::to_string(N) + " < " + std::to_string(size));
            for (unsigned int i=0; i<size; ++i) m_data[i] = data[i];
            return *this;
        }
        GT_FUNCTION T &operator[](unsigned int i) { return m_data[i]; }

        GT_FUNCTION T const &operator[](unsigned int i) const { return m_data[i]; }
    };

    template<typename T, typename Array, typename Field, unsigned int N>
    __global__ void pack_kernel_u(
        pack_arg_t<int,   N> sizes, 
        pack_arg_t<T*,    N> buffers, 
        pack_arg_t<Array, N> firsts, 
        pack_arg_t<Array, N> local_strides,
        pack_arg_t<Field, N> fields)
    {
        using layout_t = typename Field::layout_map;
        const int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
        const int data_lu_index = blockIdx.y;

        const int size = sizes[data_lu_index];
        if (thread_index < size)
        {
            Array local_coordinate;
            detail::compute_coordinate<Array::size()>::template apply<layout_t>(local_strides[data_lu_index],local_coordinate,thread_index);
            // add offset
            const auto memory_coordinate = local_coordinate + firsts[data_lu_index] + fields[data_lu_index].offsets();
            // multiply with memory strides
            const auto idx = dot(memory_coordinate, fields[data_lu_index].byte_strides());
            buffers[data_lu_index][thread_index] = *reinterpret_cast<const T*>((const char*)fields[data_lu_index].data() + idx);
        }
    }

    template<typename T, typename Array, typename Field, unsigned int N>
    __global__ void unpack_kernel_u(
        pack_arg_t<int,      N> sizes, 
        pack_arg_t<const T*, N> buffers, 
        pack_arg_t<Array,    N> firsts, 
        pack_arg_t<Array,    N> local_strides,
        pack_arg_t<Field,    N> fields)
    {
        using layout_t = typename Field::layout_map;
        const int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
        const int data_lu_index = blockIdx.y;

        const int size = sizes[data_lu_index];
        if (thread_index < size)
        {
            Array local_coordinate;
            detail::compute_coordinate<Array::size()>::template apply<layout_t>(local_strides[data_lu_index],local_coordinate,thread_index);
            // add offset
            const auto memory_coordinate = local_coordinate + firsts[data_lu_index] + fields[data_lu_index].offsets();
            // multiply with memory strides
            const auto idx = dot(memory_coordinate, fields[data_lu_index].byte_strides());
            *reinterpret_cast<T*>((char*)fields[data_lu_index].data() + idx) = buffers[data_lu_index][thread_index];
        }
    }

    template<>
    struct packer<device::gpu>
    {
        template<typename T, typename FieldType, typename Map, typename Futures, typename Communicator>
        static void pack_u(Map& map, Futures& send_futures, Communicator& comm)
        {
            using send_buffer_type     = typename Map::send_buffer_type;
            using field_buffer_type    = typename send_buffer_type::field_buffer_type;
            using index_container_type = typename field_buffer_type::index_container_type;
            using dimension            = typename index_container_type::value_type::dimension;
            using array_t              = array<int, dimension::value>;

            static std::vector<array_t> firsts;
            static std::vector<array_t> lasts;
            static std::vector<int>     sizes;
            static std::vector<T*> buffer_offsets;
            static std::vector<FieldType> fields;

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
            const int block_size = 128;
            num_streams = 0;
            for (auto& p0 : map.send_memory)
            {
                for (auto& p1: p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        firsts.clear();
                        lasts.clear();
                        sizes.clear();
                        buffer_offsets.clear();
                        fields.clear();
                        int num_blocks_y = 0;
                        int max_size = 0;
                        for (const auto& fb : p1.second.field_buffers)
                        {
                            T* buffer_address = reinterpret_cast<T*>(p1.second.buffer.data()+fb.offset);
                            for (const auto& it_space_pair : *fb.index_container)
                            {
                                ++num_blocks_y;
                                const int size = it_space_pair.size();
                                max_size = std::max(size,max_size);
                                array_t first, last;
                                std::copy(&it_space_pair.local().first()[0], &it_space_pair.local().first()[dimension::value], first.data());
                                std::copy(&it_space_pair.local().last()[0],  &it_space_pair.local().last() [dimension::value], last.data());
                                firsts.push_back(first);
                                buffer_offsets.push_back(buffer_address);
                                buffer_address += size;
                                sizes.push_back(size);
                                fields.push_back(*reinterpret_cast<FieldType*>(fb.field_ptr));
                                array_t local_extents, local_strides;
                                for (std::size_t i=0; i<dimension::value; ++i)  
                                    local_extents[i] = 1 + last[i] - first[i];
                                detail::compute_strides<dimension::value>::template apply<typename FieldType::layout_map>(local_extents, local_strides);
                                lasts.push_back(local_strides);
                            }
                        }
                        const int num_blocks_x = (max_size+block_size-1)/block_size;
                        unsigned int count = 0;
                        while (num_blocks_y)
                        {
                            if (num_blocks_y > 36)
                            {
                                dim3 dimBlock(block_size, 1);
                                dim3 dimGrid(num_blocks_x, 36);
                                pack_kernel_u<T><<<dimGrid, dimBlock, 0, streams[num_streams]>>>(
                                    pack_arg_t<int,       36>().fill(sizes.data()+count, 36),
                                    pack_arg_t<T*,        36>().fill(buffer_offsets.data()+count, 36),
                                    pack_arg_t<array_t,   36>().fill(firsts.data()+count, 36),
                                    pack_arg_t<array_t,   36>().fill(lasts.data()+count, 36),
                                    pack_arg_t<FieldType, 36>().fill(fields.data()+count, 36)
                                );
                                count += 36;
                                num_blocks_y -= 36;
                            }
                            else 
                            {
                                dim3 dimBlock(block_size, 1);
                                dim3 dimGrid(num_blocks_x, num_blocks_y);
                                if (num_blocks_y < 7)
                                {
                                    pack_kernel_u<T><<<dimGrid, dimBlock, 0, streams[num_streams]>>>(
                                        pack_arg_t<int,       6>().fill(sizes.data()+count,          num_blocks_y),
                                        pack_arg_t<T*,        6>().fill(buffer_offsets.data()+count, num_blocks_y),
                                        pack_arg_t<array_t,   6>().fill(firsts.data()+count,         num_blocks_y),
                                        pack_arg_t<array_t,   6>().fill(lasts.data()+count,          num_blocks_y),
                                        pack_arg_t<FieldType, 6>().fill(fields.data()+count,         num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 13)
                                {
                                    pack_kernel_u<T><<<dimGrid, dimBlock, 0, streams[num_streams]>>>(
                                        pack_arg_t<int,      12>().fill(sizes.data()+count,          num_blocks_y),
                                        pack_arg_t<T*,       12>().fill(buffer_offsets.data()+count, num_blocks_y),
                                        pack_arg_t<array_t,  12>().fill(firsts.data()+count,         num_blocks_y),
                                        pack_arg_t<array_t,  12>().fill(lasts.data()+count,          num_blocks_y),
                                        pack_arg_t<FieldType,12>().fill(fields.data()+count,         num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 25)
                                {
                                    pack_kernel_u<T><<<dimGrid, dimBlock, 0, streams[num_streams]>>>(
                                        pack_arg_t<int,      24>().fill(sizes.data()+count,          num_blocks_y),
                                        pack_arg_t<T*,       24>().fill(buffer_offsets.data()+count, num_blocks_y),
                                        pack_arg_t<array_t,  24>().fill(firsts.data()+count,         num_blocks_y),
                                        pack_arg_t<array_t,  24>().fill(lasts.data()+count,          num_blocks_y),
                                        pack_arg_t<FieldType,24>().fill(fields.data()+count,         num_blocks_y)
                                    );
                                }
                                else
                                {
                                    pack_kernel_u<T><<<dimGrid, dimBlock, 0, streams[num_streams]>>>(
                                        pack_arg_t<int,      36>().fill(sizes.data()+count,          num_blocks_y),
                                        pack_arg_t<T*,       36>().fill(buffer_offsets.data()+count, num_blocks_y),
                                        pack_arg_t<array_t,  36>().fill(firsts.data()+count,         num_blocks_y),
                                        pack_arg_t<array_t,  36>().fill(lasts.data()+count,          num_blocks_y),
                                        pack_arg_t<FieldType,36>().fill(fields.data()+count,         num_blocks_y)
                                    );
                                }
                                count += num_blocks_y;
                                num_blocks_y = 0;
                            }
                        }
                        ++num_streams;
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
                        cudaStreamDestroy(streams[num_streams]);
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
        }

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
                        {
                            fb.call_back( p1.second.buffer.data() + fb.offset, *fb.index_container, (void*)(&streams[num_streams]));
                        }
                        ++num_streams;
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

        template<typename T, typename FieldType, typename BufferMem>
        static void unpack_u(BufferMem& m)
        {
            using recv_buffer_type     = typename BufferMem::recv_buffer_type;
            using field_buffer_type    = typename recv_buffer_type::field_buffer_type;
            using index_container_type = typename field_buffer_type::index_container_type;
            using dimension            = typename index_container_type::value_type::dimension;
            using array_t              = array<int, dimension::value>;

            static std::vector<array_t> firsts;
            static std::vector<array_t> lasts;
            static std::vector<int>     sizes;
            static std::vector<const T*> buffer_offsets;
            static std::vector<FieldType> fields;

            std::vector<cudaStream_t> streams(m.m_recv_futures.size());
            std::vector<std::size_t> index_list(m.m_recv_futures.size());
            std::size_t i = 0;
            for (auto& x : streams)
            {
                cudaStreamCreate(&x);
                index_list[i] = i;
                ++i;
            }
            const int block_size = 128;
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
                        //for (const auto& fb : *m.m_recv_hooks[k].second)
                        //    fb.call_back(m.m_recv_hooks[k].first + fb.offset, *fb.index_container, (void*)(&streams[k]));
                        
                        firsts.clear();
                        lasts.clear();
                        sizes.clear();
                        buffer_offsets.clear();
                        fields.clear();
                        int num_blocks_y = 0;
                        int max_size = 0;
                        for (const auto& fb : *m.m_recv_hooks[k].second)
                        {
                            const T* buffer_address = reinterpret_cast<const T*>(m.m_recv_hooks[k].first+fb.offset);
                            for (const auto& it_space_pair : *fb.index_container)
                            {
                                ++num_blocks_y;
                                const int size = it_space_pair.size();
                                max_size = std::max(size,max_size);
                                array_t first, last;
                                std::copy(&it_space_pair.local().first()[0], &it_space_pair.local().first()[dimension::value], first.data());
                                std::copy(&it_space_pair.local().last()[0],  &it_space_pair.local().last() [dimension::value], last.data());
                                firsts.push_back(first);
                                buffer_offsets.push_back(buffer_address);
                                buffer_address += size;
                                sizes.push_back(size);
                                fields.push_back(*reinterpret_cast<FieldType*>(fb.field_ptr));
                                array_t local_extents, local_strides;
                                for (std::size_t i=0; i<dimension::value; ++i)  
                                    local_extents[i] = 1 + last[i] - first[i];
                                detail::compute_strides<dimension::value>::template apply<typename FieldType::layout_map>(local_extents, local_strides);
                                lasts.push_back(local_strides);
                            }
                        }
                        const int num_blocks_x = (max_size+block_size-1)/block_size;
                        unsigned int count = 0;
                        while (num_blocks_y)
                        {
                            if (num_blocks_y > 36)
                            {
                                dim3 dimBlock(block_size, 1);
                                dim3 dimGrid(num_blocks_x, 36);
                                unpack_kernel_u<T><<<dimGrid, dimBlock, 0, streams[k]>>>(
                                    pack_arg_t<int,       36>().fill(sizes.data()+count, 36),
                                    pack_arg_t<const T*,  36>().fill(buffer_offsets.data()+count, 36),
                                    pack_arg_t<array_t,   36>().fill(firsts.data()+count, 36),
                                    pack_arg_t<array_t,   36>().fill(lasts.data()+count, 36),
                                    pack_arg_t<FieldType, 36>().fill(fields.data()+count, 36)
                                );
                                count += 36;
                                num_blocks_y -= 36;
                            }
                            else 
                            {
                                dim3 dimBlock(block_size, 1);
                                dim3 dimGrid(num_blocks_x, num_blocks_y);
                                if (num_blocks_y < 7)
                                {
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, streams[k]>>>(
                                        pack_arg_t<int,       6>().fill(sizes.data()+count,          num_blocks_y),
                                        pack_arg_t<const T*,  6>().fill(buffer_offsets.data()+count, num_blocks_y),
                                        pack_arg_t<array_t,   6>().fill(firsts.data()+count,         num_blocks_y),
                                        pack_arg_t<array_t,   6>().fill(lasts.data()+count,          num_blocks_y),
                                        pack_arg_t<FieldType, 6>().fill(fields.data()+count,         num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 13)
                                {
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, streams[k]>>>(
                                        pack_arg_t<int,      12>().fill(sizes.data()+count,          num_blocks_y),
                                        pack_arg_t<const T*, 12>().fill(buffer_offsets.data()+count, num_blocks_y),
                                        pack_arg_t<array_t,  12>().fill(firsts.data()+count,         num_blocks_y),
                                        pack_arg_t<array_t,  12>().fill(lasts.data()+count,          num_blocks_y),
                                        pack_arg_t<FieldType,12>().fill(fields.data()+count,         num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 25)
                                {
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, streams[k]>>>(
                                        pack_arg_t<int,      24>().fill(sizes.data()+count,          num_blocks_y),
                                        pack_arg_t<const T*, 24>().fill(buffer_offsets.data()+count, num_blocks_y),
                                        pack_arg_t<array_t,  24>().fill(firsts.data()+count,         num_blocks_y),
                                        pack_arg_t<array_t,  24>().fill(lasts.data()+count,          num_blocks_y),
                                        pack_arg_t<FieldType,24>().fill(fields.data()+count,         num_blocks_y)
                                    );
                                }
                                else
                                {
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, streams[k]>>>(
                                        pack_arg_t<int,      36>().fill(sizes.data()+count,          num_blocks_y),
                                        pack_arg_t<const T*, 36>().fill(buffer_offsets.data()+count, num_blocks_y),
                                        pack_arg_t<array_t,  36>().fill(firsts.data()+count,         num_blocks_y),
                                        pack_arg_t<array_t,  36>().fill(lasts.data()+count,          num_blocks_y),
                                        pack_arg_t<FieldType,36>().fill(fields.data()+count,         num_blocks_y)
                                    );
                                }
                                count += num_blocks_y;
                                num_blocks_y = 0;
                            }
                        }
                    }
                }
            }
            for (auto& x : streams) 
            {
                cudaStreamSynchronize(x);
                cudaStreamDestroy(x);
            }
        }

        template<typename BufferMem>
        static void unpack(BufferMem& m)
        {
            //auto policy = std::launch::async;
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
                        for (const auto& fb : *m.m_recv_hooks[k].second)
                            fb.call_back(m.m_recv_hooks[k].first + fb.offset, *fb.index_container, (void*)(&streams[k]));
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


#ifdef GHEX_COMM_2_TIMINGS
        template<typename BufferMem,typename Timings>
        static void unpack(BufferMem& m, Timings& t)
        {
            //auto policy = std::launch::async;
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
                        const auto t0 = Timings::clock_type::now(); 
                        for (const auto& fb : *m.m_recv_hooks[k].second)
                            fb.call_back(m.m_recv_hooks[k].first + fb.offset, *fb.index_container, (void*)(&streams[k]));
                        const auto t1 = Timings::clock_type::now(); 
                        t.unpack_launch_time += t1-t0;
                    }
                }
            }
            const auto t0 = Timings::clock_type::now(); 
            //int k = 0;
            for (auto& x : streams) 
            {
                //futures[k++].get();
                cudaStreamSynchronize(x);
                cudaStreamDestroy(x);
            }
            const auto t1 = Timings::clock_type::now(); 
            t.unpack_launch_time += t1-t0;
        }
#endif
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
        using pattern_type            = pattern<P,GridType,DomainIdType>;
        using pattern_container_type  = pattern_container<P,GridType,DomainIdType>;
        using this_type               = communication_object<P,GridType,DomainIdType>;

        template<typename D, typename F>
        using buffer_info_type        = buffer_info<pattern_type,D,F>;

    private: // member types
        using communicator_type       = typename handle_type::communicator_type;
        using address_type            = typename communicator_type::address_type;
        using index_container_type    = typename pattern_type::index_container_type;
        using pack_function_type      = std::function<void(void*,const index_container_type&, void*)>;
        using unpack_function_type    = std::function<void(const void*,const index_container_type&, void*)>;

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
            using index_container_type = typename pattern_type::map_type::mapped_type;
            Function call_back;
            const index_container_type* index_container;
            std::size_t offset;
            void* field_ptr;
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

        communication_object()
        : m_valid(false) {}

        communication_object(const communication_object&) = delete;
        communication_object(communication_object&&) = default;

    public: // member functions

        auto& tag_map() noexcept { return m_max_tag_map; }
        const auto& tag_map() const noexcept { return m_max_tag_map; }

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
            handle_type h(*this,std::get<0>(buffer_info_tuple)->get_pattern().communicator(), [this](){this->wait();});
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
                    true, field_ptr);
                this->allocate<device_type,value_type,typename buffer_memory<device_type>::send_buffer_type>(
                    mem->send_memory[bi->device_id()],
                    bi->get_pattern().send_halos(),
                    [field_ptr](void* buffer, const index_container_type& c, void* arg) 
                        { field_ptr->pack(reinterpret_cast<value_type*>(buffer),c,arg); },
                    my_dom_id,
                    bi->device_id(),
                    tag_offsets[i],
                    false, field_ptr);
                ++i;
            });
            post(h.m_comm);
            pack(h.m_comm);
            return h; 
        }

        template<typename Device, typename Field>
        [[nodiscard]] handle_type exchange(buffer_info_type<Device,Field>* first, std::size_t length)
        {
            auto h = exchange_impl(first, length);
            pack(h.m_comm);
            return h;
        }

#ifdef __CUDACC__
        template<typename Device, typename T, int... Order>
        [[nodiscard]] std::enable_if_t<std::is_same<Device,device::gpu>::value, handle_type>
        exchange_u(
            buffer_info_type<
                Device,
                simple_field_wrapper<
                    T,
                    Device,
                    structured_domain_descriptor<
                        domain_id_type, 
                        sizeof...(Order)
                    >, 
                    Order...
                >
            >* first, std::size_t length)
        {
            using memory_t   = buffer_memory<device::gpu>;
            using field_type = std::remove_reference_t<decltype(first->get_field())>;
            using value_type = typename field_type::value_type;
            auto h = exchange_impl(first, length);
            //h.m_wait_fct = &this_type::wait<value_type,field_type>;
            h.m_wait_fct = [this](){this->wait_u<value_type,field_type>();};
            memory_t& mem = std::get<memory_t>(m_mem);
            packer<device::gpu>::pack_u<value_type,field_type>(mem, m_send_futures, h.m_comm);
            return h;
        }
#endif
        template<typename Device, typename T, int... Order>
        [[nodiscard]] std::enable_if_t<std::is_same<Device,device::cpu>::value, handle_type>
        exchange_u(
            buffer_info_type<
                Device,
                simple_field_wrapper<
                    T,
                    Device,
                    structured_domain_descriptor<
                        domain_id_type, 
                        sizeof...(Order)
                    >, 
                    Order...
                >
            >* first, std::size_t length)
        {
            return exchange(first, length);
        }

    private:
        template<typename Device, typename Field>
        [[nodiscard]] handle_type exchange_impl(buffer_info_type<Device,Field>* first, std::size_t length)
        {
            if (m_valid) throw std::runtime_error("earlier exchange operation was not finished");
            m_valid = true;
            
            using memory_t               = buffer_memory<Device>*;
            using value_type             = typename buffer_info_type<Device,Field>::value_type;

            handle_type h(*this, first->get_pattern().communicator(), [this](){this->wait();});
            memory_t mem{&(std::get<buffer_memory<Device>>(m_mem))};
            std::vector<int> tag_offsets(length);
            for (std::size_t k=0; k<length; ++k)
                tag_offsets[k] = m_max_tag_map[&((first+k)->get_pattern_container())];

            for (std::size_t k=0; k<length; ++k)
            {
                auto field_ptr = &((first+k)->get_field());
                const auto my_dom_id  =(first+k)->get_field().domain_id();

                this->allocate<Device,value_type,typename buffer_memory<Device>::recv_buffer_type>(
                    mem->recv_memory[(first+k)->device_id()],
                    (first+k)->get_pattern().recv_halos(),
                    [field_ptr](const void* buffer, const index_container_type& c, void* arg) 
                        { field_ptr->unpack(reinterpret_cast<const value_type*>(buffer),c,arg); },
                    my_dom_id,
                    (first+k)->device_id(),
                    tag_offsets[k],
                    true, field_ptr);
                this->allocate<Device,value_type,typename buffer_memory<Device>::send_buffer_type>(
                    mem->send_memory[(first+k)->device_id()],
                    (first+k)->get_pattern().send_halos(),
                    [field_ptr](void* buffer, const index_container_type& c, void* arg) 
                        { field_ptr->pack(reinterpret_cast<value_type*>(buffer),c,arg); },
                    my_dom_id,
                    (first+k)->device_id(),
                    tag_offsets[k],
                    false, field_ptr);
            }
            post(h.m_comm);
            return h;
        }

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
        }

        void pack(communicator_type& comm)
        {
            detail::for_each(m_mem, [this,&comm](auto& m)
            {
                using device_type = typename std::remove_reference_t<decltype(m)>::device_type;
                packer<device_type>::pack(m,m_send_futures,comm);
            });
        }

#ifdef __CUDACC__
        template<typename T, typename Field>
        void wait_u()
        {
            if (!m_valid) return;
            
            using memory_t   = buffer_memory<device::gpu>;
            memory_t& mem = std::get<memory_t>(m_mem);

            packer<device::gpu>::unpack_u<T,Field>(mem);

            for (auto& f : m_send_futures) 
                f.wait();

            clear();
        }
#endif

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

            detail::for_each(m_mem, [this, &t](auto& m)
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
        template<typename Device, typename ValueType, typename BufferType, typename Memory, typename Halos, typename Function, typename DeviceIdType, typename Field = void>
        void allocate(Memory& memory, const Halos& halos, Function&& func, domain_id_type my_dom_id, DeviceIdType device_id, 
                      int tag_offset, bool receive, Field* field_ptr = nullptr)
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
                    /*std::cout << "field_ptr = " << field_ptr << ", data = " << field_ptr->data() << std::endl;
                    ValueType first_element;
                    cudaMemcpy(&first_element, field_ptr, sizeof(ValueType), cudaMemcpyDeviceToHost);
                    //ValueType first_p_element;
                    //cudaMemcpy(&first_p_element, field_ptr+195, sizeof(ValueType), cudaMemcpyDeviceToHost);
                    std::cout << "     first_element   =  " << first_element << std::endl;                                 
                    //std::cout << "     first_p_element =  " << first_p_element << std::endl;*/
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
                    typename BufferType::field_buffer_type{std::forward<Function>(func), &p_id_c.second, prev_size + padding, field_ptr});
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

