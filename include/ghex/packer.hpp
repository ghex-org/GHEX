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
#ifndef INCLUDED_GHEX_PACKER_HPP
#define INCLUDED_GHEX_PACKER_HPP

#include "./common/await_futures.hpp"
#include "./arch_list.hpp"
#include "./structured/field_utils.hpp"
#include "./cuda_utils/kernel_argument.hpp"
#include "./cuda_utils/future.hpp"
#include <gridtools/common/array.hpp>

namespace gridtools {

    namespace ghex {

        /** @brief generic implementation of pack and unpack */
        template<typename Arch>
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
                            send_futures.push_back(comm.send(p1.second.buffer, p1.second.address, p1.second.tag));
                        }
                    }
                }
            }

            template<typename BufferMem>
            static void unpack(BufferMem& m)
            {
                await_futures(
                    m.m_recv_futures,
                    [](typename BufferMem::hook_type hook)
                    {
                        for (const auto& fb :  hook->field_infos)
                            fb.call_back(hook->buffer.data() + fb.offset, *fb.index_container, nullptr);
                    });
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
            cuda::kernel_argument<kernel_args<T,Array,Field>, N> args)
        {
            using layout_t = typename Field::layout_map;
            const int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
            const int data_lu_index = blockIdx.y;

            const auto& arg = args[data_lu_index];
            const int size = arg.size;
            if (thread_index < size)
            {
                Array local_coordinate;
                structured::detail::compute_coordinate<Array::size()>::template apply<layout_t>(arg.strides,local_coordinate,thread_index);
                // add offset
                const auto memory_coordinate = local_coordinate + arg.first + arg.field.offsets();
                // multiply with memory strides
                const auto idx = dot(memory_coordinate, arg.field.byte_strides());
                arg.buffer[thread_index] = *reinterpret_cast<const T*>((const char*)arg.field.data() + idx);
            }
        }

        template<typename T, typename Array, typename Field, unsigned int N>
        __global__ void unpack_kernel_u(
            cuda::kernel_argument<kernel_args<T,Array,Field>, N> args)
        {
            using layout_t = typename Field::layout_map;
            const int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
            const int data_lu_index = blockIdx.y;

            const auto& arg = args[data_lu_index];
            const int size = arg.size;
            if (thread_index < size)
            {
                Array local_coordinate;
                structured::detail::compute_coordinate<Array::size()>::template apply<layout_t>(arg.strides,local_coordinate,thread_index);
                // add offset
                const auto memory_coordinate = local_coordinate + arg.first + arg.field.offsets();
                // multiply with memory strides
                const auto idx = dot(memory_coordinate, arg.field.byte_strides());
                *reinterpret_cast<T*>((char*)arg.field.data() + idx) = arg.buffer[thread_index];
            }
        }

        /** @brief specialization for gpus, including vector interface special functions */
        template<>
        struct packer<gpu>
        {
            template<typename Map, typename Futures, typename Communicator>
            static void pack(Map& map, Futures& send_futures,Communicator& comm)
            {
                using send_buffer_type     = typename Map::send_buffer_type;
                using future_type = cuda::future<send_buffer_type*>;
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
                std::vector<future_type> stream_futures;
                stream_futures.reserve(num_streams);
                num_streams = 0;
                for (auto& p0 : map.send_memory)
                {
                    for (auto& p1: p0.second)
                    {
                        if (p1.second.size > 0u)
                        {
                            for (const auto& fb : p1.second.field_infos)
                            {
                                fb.call_back( p1.second.buffer.data() + fb.offset, *fb.index_container, (void*)(&p1.second.m_cuda_stream.get()));
                            }
                            stream_futures.push_back( future_type{&(p1.second), p1.second.m_cuda_stream} );
                            ++num_streams;
                        }
                    }
                }
                await_futures(
                    stream_futures, 
                    [&comm,&send_futures](send_buffer_type* b)
                    {
                        send_futures.push_back(comm.send(b->buffer, b->address, b->tag));
                    });
            }

            template<typename BufferMem>
            static void unpack(BufferMem& m)
            {
                std::vector<cudaStream_t*> stream_ptrs;
                stream_ptrs.reserve(m.m_recv_futures.size());
                await_futures(
                    m.m_recv_futures,
                    [&stream_ptrs](typename BufferMem::hook_type hook)
                    {
                        auto stream_ptr = &hook->m_cuda_stream.get();
                        for (const auto& fb : hook->field_infos)
                                fb.call_back(hook->buffer.data() + fb.offset, *fb.index_container, (void*)(stream_ptr));
                        stream_ptrs.push_back(stream_ptr);

                    });
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
                            ++num_streams;
                        }
                    }
                }

                using future_type = cuda::future<send_buffer_type*>;
                std::vector<future_type> stream_futures;
                stream_futures.reserve(num_streams);

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
                                    structured::detail::compute_strides<dimension::value>::template apply<typename FieldType::layout_map>(local_extents, local_strides);
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
                                    pack_kernel_u<T><<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream>>>(
                                        cuda::make_kernel_arg<36>(args.data()+count, 36)
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
                                        pack_kernel_u<T><<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream>>>(
                                            cuda::make_kernel_arg< 6>(args.data()+count, num_blocks_y)
                                        );
                                    }
                                    else if (num_blocks_y < 13)
                                    {
                                        pack_kernel_u<T><<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream>>>(
                                            cuda::make_kernel_arg<12>(args.data()+count, num_blocks_y)
                                        );
                                    }
                                    else if (num_blocks_y < 25)
                                    {
                                        pack_kernel_u<T><<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream>>>(
                                            cuda::make_kernel_arg<24>(args.data()+count, num_blocks_y)
                                        );
                                    }
                                    else
                                    {
                                        pack_kernel_u<T><<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream>>>(
                                            cuda::make_kernel_arg<36>(args.data()+count, num_blocks_y)
                                        );
                                    }
                                    count += num_blocks_y;
                                    num_blocks_y = 0;
                                }
                            }
                            stream_futures.push_back( future_type{&(p1.second), p1.second.m_cuda_stream} );
                            ++num_streams;
                        }
                    }
                }
                await_futures(
                    stream_futures, 
                    [&comm,&send_futures](send_buffer_type* b)
                    {
                        send_futures.push_back(comm.send(b->buffer, b->address, b->tag));
                    });
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
                const int block_size = 128;

                std::vector<arg_t> args;
                args.reserve(64);

                std::vector<cudaStream_t*> stream_ptrs;
                stream_ptrs.reserve(m.m_recv_futures.size());
                await_futures(
                    m.m_recv_futures,
                    [&block_size,&stream_ptrs,&args](typename BufferMem::hook_type hook)
                    {
                        auto stream_ptr = &hook->m_cuda_stream.get();
                        args.resize(0);
                        int num_blocks_y = 0;
                        int max_size = 0;
                        for (const auto& fb : hook->field_infos)
                        {
                            T* buffer_address = reinterpret_cast<T*>(hook->buffer.data()+fb.offset);
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
                                structured::detail::compute_strides<dimension::value>::template apply<typename FieldType::layout_map>(local_extents, local_strides);
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
                                unpack_kernel_u<T><<<dimGrid, dimBlock, 0, *stream_ptr>>>(
                                    cuda::make_kernel_arg<36>(args.data()+count, 36)
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
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, *stream_ptr>>>(
                                        cuda::make_kernel_arg<6>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 13)
                                {
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, *stream_ptr>>>(
                                        cuda::make_kernel_arg<12>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 25)
                                {
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, *stream_ptr>>>(
                                        cuda::make_kernel_arg<24>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else
                                {
                                    unpack_kernel_u<T><<<dimGrid, dimBlock, 0, *stream_ptr>>>(
                                        cuda::make_kernel_arg<36>(args.data()+count, num_blocks_y)
                                    );
                                }
                                count += num_blocks_y;
                                num_blocks_y = 0;
                            }
                        }
                        stream_ptrs.push_back(stream_ptr);
                    });
                for (auto x : stream_ptrs) 
                {
                    cudaStreamSynchronize(*x);
                }
            }
        };
#endif

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_PACKER_HPP */

