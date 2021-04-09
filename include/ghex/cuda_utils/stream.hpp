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
#ifndef INCLUDED_GHEX_CUDA_STREAM_HPP
#define INCLUDED_GHEX_CUDA_STREAM_HPP

#include "../common/c_managed_struct.hpp"
#include "./error.hpp"
#include <memory>

#include "../common/defs.hpp"
#ifdef GHEX_CUDACC
#include "../common/cuda_runtime.hpp"
#endif

namespace gridtools {

    namespace ghex {

        namespace cuda {

#ifdef GHEX_CUDACC

            /** @brief thin wrapper around a cuda stream */
            struct stream
            {
                
                GHEX_C_MANAGED_STRUCT(stream_type, cudaStream_t, cudaStreamCreateWithFlags, cudaStreamDestroy)

                stream_type m_stream;

                stream()
                : m_stream{cudaStreamNonBlocking}
                {} 
                stream(const stream&) = delete;
                stream& operator=(const stream&) = delete;
                stream(stream&& other) = default;
                stream& operator=(stream&&) = default;
        
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
            struct stream
            {
                // default construct
                stream() {}
                stream(bool) {}

                // non-copyable
                stream(const stream&) noexcept = delete;
                stream& operator=(const stream&)= delete;

                // movable
                stream(stream&& other) noexcept =  default;
                stream& operator=(stream&&) noexcept = default;
            };
#endif

        } // namespace cua

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_CUDA_STREAM_HPP */

