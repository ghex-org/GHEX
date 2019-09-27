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
#ifndef INCLUDED_CUDA_STREAM_HPP
#define INCLUDED_CUDA_STREAM_HPP

#include "../common/c_managed_struct.hpp"
#include "./error.hpp"
#include <memory>

namespace gridtools {
    namespace ghex {

#ifdef __CUDACC__

        struct cuda_stream
        {
            
            GHEX_C_MANAGED_STRUCT(stream_type, cudaStream_t, cudaStreamCreateWithFlags, cudaStreamDestroy)

            stream_type m_stream;

            cuda_stream()
            : m_stream{cudaStreamNonBlocking}
            {} 
            cuda_stream(const cuda_stream&) = delete;
            cuda_stream& operator=(const cuda_stream&) = delete;
            cuda_stream(cuda_stream&& other) = default;
            cuda_stream& operator=(cuda_stream&&) = default;
    
            operator bool() const noexcept {return (bool)m_stream;}
            operator       cudaStream_t&()       noexcept {return m_stream;}
            operator const cudaStream_t&() const noexcept {return m_stream;}
                  cudaStream_t& get()       noexcept {return m_stream;}
            const cudaStream_t& get() const noexcept {return m_stream;}

            void sync()
            {
                GHEX_CHECK_CUDA_RESULT( cudaStreamSynchronize(m_stream) );
            }
            
        };
#else
        struct cuda_stream
        {
            // default construct
            cuda_stream() {}
            cuda_stream(bool) {}

            // non-copyable
            cuda_stream(const cuda_stream&) noexcept = delete;
            cuda_stream& operator=(const cuda_stream&)= delete;

            // movable
            cuda_stream(cuda_stream&& other) noexcept =  default;
            cuda_stream& operator=(cuda_stream&&) noexcept = default;
        };
#endif

    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_CUDA_STREAM_HPP */

