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
#ifndef INCLUDED_PACKER_HPP
#define INCLUDED_PACKER_HPP

#include "./devices.hpp"
#include "./structured/field_utils.hpp"
#include "./cuda_utils/kernel_argument.hpp"
#include <gridtools/common/array.hpp>

namespace gridtools {

    /** @brief generic implementation of pack and unpack */
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
                        for (const auto& fb : p1.second.field_infos)
                            fb.call_back( p1.second.buffer.data() + fb.offset, *fb.index_container, nullptr);
                        send_futures.push_back(comm.isend(
                            p1.second.address,
                            p1.second.tag,
                            p1.second.buffer));
                    }
                }
            }
        }

        template<typename BufferMem>
        static void unpack(BufferMem& m)
        {
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
                                for (const auto& fb : *m.m_recv_hooks[k].second)
                                    fb.call_back(m.m_recv_hooks[k].first + fb.offset, *fb.index_container,nullptr);
                    }
                }
            }
        }
    };


	
#ifdef __CUDACC__
    
    template<typename T, typename Array, typename Field>
    struct kernel_args
    {
        int   size;
        T* buffer;
        Array first;
        Array strides;
        Field field;
    };

    template<typename T, typename Array, typename Field, unsigned int N>
    __global__ void pack_kernel_u(
        kernel_argument<kernel_args<T,Array,Field>, N> args)
    {
        using layout_t = typename Field::layout_map;
        const int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
        const int data_lu_index = blockIdx.y;

        const auto& arg = args[data_lu_index];
        const int size = arg.size;
        if (thread_index < size)
        {
            Array local_coordinate;
            detail::compute_coordinate<Array::size()>::template apply<layout_t>(arg.strides,local_coordinate,thread_index);
            // add offset
            const auto memory_coordinate = local_coordinate + arg.first + arg.field.offsets();
            // multiply with memory strides
            const auto idx = dot(memory_coordinate, arg.field.byte_strides());
            arg.buffer[thread_index] = *reinterpret_cast<const T*>((const char*)arg.field.data() + idx);
        }
    }

    template<typename T, typename Array, typename Field, unsigned int N>
    __global__ void unpack_kernel_u(
        kernel_argument<kernel_args<T,Array,Field>, N> args)
    {
        using layout_t = typename Field::layout_map;
        const int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
        const int data_lu_index = blockIdx.y;

        const auto& arg = args[data_lu_index];
        const int size = arg.size;
        if (thread_index < size)
        {
            Array local_coordinate;
            detail::compute_coordinate<Array::size()>::template apply<layout_t>(arg.strides,local_coordinate,thread_index);
            // add offset
            const auto memory_coordinate = local_coordinate + arg.first + arg.field.offsets();
            // multiply with memory strides
            const auto idx = dot(memory_coordinate, arg.field.byte_strides());
            *reinterpret_cast<T*>((char*)arg.field.data() + idx) = arg.buffer[thread_index];
        }
    }

    /** @brief specialization for gpus, including vector interface special functions */
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
                        p1.second.m_cuda_stream.activate();
                        ++num_streams;
                    }
                }
            }
            num_streams = 0;
            for (auto& p0 : map.send_memory)
            {
                for (auto& p1: p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        for (const auto& fb : p1.second.field_infos)
                        {
                            fb.call_back( p1.second.buffer.data() + fb.offset, *fb.index_container, (void*)(&p1.second.m_cuda_stream.m_stream));
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
                        p1.second.m_cuda_stream.sync();
                        send_futures.push_back(comm.isend(
                            p1.second.address,
                            p1.second.tag,
                            p1.second.buffer));
                        ++num_streams;
                    }
                }
            }
        }

        template<typename BufferMem>
        static void unpack(BufferMem& m)
        {
            std::vector<cudaStream_t*> stream_ptrs(m.m_recv_futures.size());
            std::vector<std::size_t> index_list(m.m_recv_futures.size());
            std::size_t i = 0;
            for (auto& p0 : m.recv_memory)
            {
                for (auto& p1: p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        p1.second.m_cuda_stream.activate();
                        stream_ptrs[i] = &(p1.second.m_cuda_stream.m_stream);
                        index_list[i] = i;
                        ++i;
                    }
                }
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
                        for (const auto& fb : *m.m_recv_hooks[k].second)
                            fb.call_back(m.m_recv_hooks[k].first + fb.offset, *fb.index_container, (void*)(stream_ptrs[k]));
                    }
                }
            }
            for (auto x : stream_ptrs) 
            {
                cudaStreamSynchronize(*x);
            }
        }

        template<typename T, typename FieldType, typename Map, typename Futures, typename Communicator>
        static void pack_u(Map& map, Futures& send_futures, Communicator& comm)
        {
            using send_buffer_type     = typename Map::send_buffer_type;
            using field_info_type      = typename send_buffer_type::field_info_type;
            using index_container_type = typename field_info_type::index_container_type;
            using dimension            = typename index_container_type::value_type::dimension;
            using array_t              = array<int, dimension::value>;

            using arg_t = kernel_args<T,array_t,FieldType>;
            std::vector<arg_t> args;
            args.reserve(64);

            std::size_t num_streams = 0;
            for (auto& p0 : map.send_memory)
            {
                for (auto& p1: p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        p1.second.buffer.resize(p1.second.size);
                        p1.second.m_cuda_stream.activate();
                        ++num_streams;
                    }
                }
            }
            const int block_size = 128;
            num_streams = 0;
            for (auto& p0 : map.send_memory)
            {
                for (auto& p1: p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        args.resize(0);
                        int num_blocks_y = 0;
                        int max_size = 0;
                        for (const auto& fb : p1.second.field_infos)
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
                                array_t local_extents, local_strides;
                                for (std::size_t i=0; i<dimension::value; ++i)  
                                    local_extents[i] = 1 + last[i] - first[i];
                                detail::compute_strides<dimension::value>::template apply<typename FieldType::layout_map>(local_extents, local_strides);
                                args.push_back( arg_t{size, buffer_address, first, local_strides, *reinterpret_cast<FieldType*>(fb.field_ptr)} );
                                buffer_address += size;
                            }
                        }
                        const int num_blocks_x = (max_size+block_size-1)/block_size;
                        // unroll kernels: can fit at most 36 arguments as pack kernel argument
                        // invoke new kernels until all data is packed
                        unsigned int count = 0;
                        while (num_blocks_y)
                        {
                            if (num_blocks_y > 36)
                            {
                                dim3 dimBlock(block_size, 1);
                                dim3 dimGrid(num_blocks_x, 36);
                                pack_kernel_u<T><<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream.m_stream>>>(
                                    make_kernel_arg<36>(args.data()+count, 36)
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
                                    pack_kernel_u<T><<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream.m_stream>>>(
                                        make_kernel_arg< 6>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 13)
                                {
                                    pack_kernel_u<T><<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream.m_stream>>>(
                                        make_kernel_arg<12>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 25)
                                {
                                    pack_kernel_u<T><<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream.m_stream>>>(
                                        make_kernel_arg<24>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else
                                {
                                    pack_kernel_u<T><<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream.m_stream>>>(
                                        make_kernel_arg<36>(args.data()+count, num_blocks_y)
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
                        p1.second.m_cuda_stream.sync();
                        send_futures.push_back(comm.isend(
                            p1.second.address,
                            p1.second.tag,
                            p1.second.buffer));
                        ++num_streams;
                    }
                }
            }
        }

        template<typename T, typename FieldType, typename BufferMem>
        static void unpack_u(BufferMem& m)
        {
            using recv_buffer_type     = typename BufferMem::recv_buffer_type;
            using field_info_type      = typename recv_buffer_type::field_info_type;
            using index_container_type = typename field_info_type::index_container_type;
            using dimension            = typename index_container_type::value_type::dimension;
            using array_t              = ::gridtools::array<int, dimension::value>;

            using arg_t = kernel_args<T,array_t,FieldType>;
            std::vector<arg_t> args;
            args.reserve(64);

            std::vector<cudaStream_t*> stream_ptrs(m.m_recv_futures.size());
            std::vector<std::size_t> index_list(m.m_recv_futures.size());
            std::size_t i = 0;
            for (auto& p0 : m.recv_memory)
            {
                for (auto& p1: p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        p1.second.m_cuda_stream.activate();
                        stream_ptrs[i] = &(p1.second.m_cuda_stream.m_stream);
                        index_list[i] = i;
                        ++i;
                    }
                }
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
                        args.resize(0);
                        int num_blocks_y = 0;
                        int max_size = 0;
                        for (const auto& fb : *m.m_recv_hooks[k].second)
                        {
                            T* buffer_address = reinterpret_cast<T*>(m.m_recv_hooks[k].first+fb.offset);
                            for (const auto& it_space_pair : *fb.index_container)
                            {
                                ++num_blocks_y;
                                const int size = it_space_pair.size();
                                max_size = std::max(size,max_size);
                                array_t first, last;
                                std::copy(&it_space_pair.local().first()[0], &it_space_pair.local().first()[dimension::value], first.data());
                                std::copy(&it_space_pair.local().last()[0],  &it_space_pair.local().last() [dimension::value], last.data());
                                array_t local_extents, local_strides;
                                for (std::size_t i=0; i<dimension::value; ++i)  
                                    local_extents[i] = 1 + last[i] - first[i];
                                detail::compute_strides<dimension::value>::template apply<typename FieldType::layout_map>(local_extents, local_strides);
                                args.push_back( arg_t{size, buffer_address, first, local_strides, *reinterpret_cast<FieldType*>(fb.field_ptr)} );
                                buffer_address += size;
                            }
                        }
                        const int num_blocks_x = (max_size+block_size-1)/block_size;
                        // unroll kernels: can fit at most 36 arguments as unpack kernel argument
                        // invoke new kernels until all data is unpacked
                        unsigned int count = 0;
                        while (num_blocks_y)
                        {
                            if (num_blocks_y > 36)
                            {
                                dim3 dimBlock(block_size, 1);
                                dim3 dimGrid(num_blocks_x, 36);
                                unpack_kernel_u<T><<<dimGrid, dimBlock, 0, *stream_ptrs[k]>>>(
                                    make_kernel_arg<36>(args.data()+count, 36)
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
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, *stream_ptrs[k]>>>(
                                        make_kernel_arg<6>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 13)
                                {
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, *stream_ptrs[k]>>>(
                                        make_kernel_arg<12>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 25)
                                {
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, *stream_ptrs[k]>>>(
                                        make_kernel_arg<24>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else
                                {
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, *stream_ptrs[k]>>>(
                                        make_kernel_arg<36>(args.data()+count, num_blocks_y)
                                    );
                                }
                                count += num_blocks_y;
                                num_blocks_y = 0;
                            }
                        }
                    }
                }
            }
            for (auto x : stream_ptrs) 
            {
                cudaStreamSynchronize(*x);
            }
        }
    };
#endif

} // namespace gridtools

#endif /* INCLUDED_PACKER_HPP */

