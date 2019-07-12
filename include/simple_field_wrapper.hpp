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
#ifndef INCLUDED_SIMPLE_FIELD_WRAPPER_HPP
#define INCLUDED_SIMPLE_FIELD_WRAPPER_HPP

#include "./structured_domain_descriptor.hpp"
#include <iostream>
#include <cstring>
#include <gridtools/common/array.hpp>
#include "devices.hpp"

namespace gridtools {

    template <typename T, size_t D>
    GT_FUNCTION
    array<T,D> operator+(array<T,D> a, const array<T,D>& b)
    {
        for (std::size_t i=0u; i<D; ++i) a[i] += b[i];
        return std::move(a);
    }
    template <typename T, size_t D>
    GT_FUNCTION
    array<T,D> operator-(array<T,D> a, const array<T,D>& b)
    {
        for (std::size_t i=0u; i<D; ++i) a[i] -= b[i];
        return std::move(a);
    }
    template <typename T, size_t D>
    GT_FUNCTION
    T dot(const array<T,D>& a, const array<T,D>& b)
    {
        T res = a[0]*b[0];
        for (std::size_t i=1u; i<D; ++i) res+=a[i]*b[i];
        return res;
    }

    namespace detail {
        template<int D, int I>
        struct compute_strides_impl
        {
            template<typename Layout, typename Coordinate>
            GT_FUNCTION
            static void apply(const Coordinate& extents, Coordinate& strides)
            {
                const auto last_idx = Layout::template find<I>();
                const auto idx      = Layout::template find<I-1>();
                strides[idx]        = strides[last_idx]*extents[last_idx];
                compute_strides_impl<D,I-1>::template apply<Layout>(extents,strides);
            }
        };
        template<int D>
        struct compute_strides_impl<D,0>
        {
            template<typename Layout, typename Coordinate>
            GT_FUNCTION
            static void apply(const Coordinate&, Coordinate&)
            {
            }
        };
        template<int D>
        struct compute_strides
        {
            template<typename Layout, typename Coordinate>
            GT_FUNCTION
            static void apply(const Coordinate& extents, Coordinate& strides)
            {
                const auto idx      = Layout::template find<D-1>();
                strides[idx]        = 1;
                compute_strides_impl<D,D-1>::template apply<Layout>(extents,strides);
            }
        };


        template<int D, int K>
        struct compute_coordinate_impl
        {
            template<typename Layout, typename Coordinate, typename I>
            GT_FUNCTION
            static void apply(const Coordinate& strides, Coordinate& coord, I i)
            {
                const auto idx = Layout::template find<D-(K)>();
                coord[idx]     = i/strides[idx];
                compute_coordinate_impl<D,K-1>::template apply<Layout>(strides, coord, i - coord[idx]*strides[idx]);
            }
        };
        template<int D>
        struct compute_coordinate_impl<D,0>
        {
            template<typename Layout, typename Coordinate, typename I>
            GT_FUNCTION
            static void apply(const Coordinate&, Coordinate&, I)
            {
            }
        };
        template<int D>
        struct compute_coordinate
        {
            template<typename Layout, typename Coordinate, typename I>
            GT_FUNCTION
            static void apply(const Coordinate& strides, Coordinate& coord, I i)
            {
                const auto idx = Layout::template find<0>();
                coord[idx]     = i/strides[idx];
                compute_coordinate_impl<D,D-1>::template apply<Layout>(strides, coord, i - coord[idx]*strides[idx]);
            }
        };
    } // namespace detail

    template<typename Device, typename Dimension, typename Layout>
    struct serialization;

    template<typename Dimension, typename Layout>
    struct serialization<device::cpu, Dimension, Layout>
    {
        template<typename T, typename IndexContainer, typename Array>
        GT_FUNCTION_HOST
        static void pack(T* buffer, const IndexContainer& c, const T* m_data, const Array& m_extents, const Array& m_offsets, const Array&, void*)
        {
            for (const auto& is : c)
            {
                detail::for_loop_pointer_arithmetic<Dimension::value,Dimension::value,Layout>::apply(
                    [m_data,buffer](auto o_data, auto o_buffer)
                    {
                        buffer[o_buffer] = m_data[o_data]; 
                    }, 
                    is.local().first(), 
                    is.local().last(),
                    m_extents,
                    m_offsets
                    );
                buffer += is.size();
            }
        }

        template<typename T, typename IndexContainer, typename Array>
        GT_FUNCTION_HOST
        static void unpack(const T* buffer, const IndexContainer& c, T* m_data, const Array& m_extents, const Array& m_offsets, const Array&, void*)
        {
            for (const auto& is : c)
            {
                detail::for_loop_pointer_arithmetic<Dimension::value,Dimension::value,Layout>::apply(
                    [m_data,buffer](auto o_data, auto o_buffer)
                    {
                        m_data[o_data] = buffer[o_buffer];
                    }, 
                    is.local().first(), 
                    is.local().last(),
                    m_extents,
                    m_offsets
                    );
                buffer += is.size();
            }
        }
    };


#ifdef __CUDACC__

    template<typename Layout, typename T, std::size_t D, typename I>
    __global__ void pack_kernel(int size, const T* data, T* buffer, 
                                array<I,D> local_first, array<I,D> local_last, 
                                array<I,D> extents, array<I,D> offsets, array<I,D> strides)
    {
        const auto index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index < size)
        {
            // compute local strides
            array<I,D> local_extents, local_strides;
            for (std::size_t i=0; i<D; ++i)  
                local_extents[i] = 1 + local_last[i] - local_first[i];
            detail::compute_strides<D>::template apply<Layout>(local_extents, local_strides);
            // compute local coordinate
            array<I,D> local_coordinate;
            detail::compute_coordinate<D>::template apply<Layout>(local_strides,local_coordinate,index);
            // add offset
            const auto memory_coordinate = local_coordinate + local_first + offsets;
            // multiply with memory strides
            const auto idx = dot(memory_coordinate, strides);
            buffer[index] = data[idx];
        }
    }
    template<typename Layout, typename T, std::size_t D, typename I>
    void pack_kernel_emulate(int blockIdx,int blockDim, int threadIdx,
            int size, const T* data, T* buffer, 
                                array<I,D> local_first, array<I,D> local_last, 
                                array<I,D> extents, array<I,D> offsets, array<I,D> strides)
    {
        const auto index = blockIdx*blockDim + threadIdx;
        if (index < size)
        {
            // compute local strides
            array<I,D> local_extents, local_strides;
            for (std::size_t i=0; i<D; ++i)  
                local_extents[i] = 1 + local_last[i] - local_first[i];
            detail::compute_strides<D>::template apply<Layout>(local_extents, local_strides);
            std::cout << "index         = " << index << std::endl;
            std::cout << "local_extents = " << local_extents[0] << ", " << local_extents[1] << ", " << local_extents[2] << std::endl;
            std::cout << "local_strides = " << local_strides[0] << ", " << local_strides[1] << ", " << local_strides[2] << std::endl;
            // compute local coordinate
            array<I,D> local_coordinate;
            detail::compute_coordinate<D>::template apply<Layout>(local_strides,local_coordinate,index);
            std::cout << "local_coord   = " << local_coordinate[0] << ", " << local_coordinate[1] << ", " << local_coordinate[2] << std::endl;
            // add offset
            const auto memory_coordinate = local_coordinate + local_first + offsets;
            std::cout << "memory_coord  = " << memory_coordinate[0] << ", " << memory_coordinate[1] << ", " << memory_coordinate[2] << std::endl;
            // multiply with memory strides
            const auto idx = dot(memory_coordinate, strides);
            std::cout << "idx           = " << idx << std::endl;
        }
    }

    template<typename Layout, typename T, std::size_t D, typename I>
    __global__ void unpack_kernel(int size, T* data, const T* buffer, 
                                array<I,D> local_first, array<I,D> local_last, 
                                array<I,D> extents, array<I,D> offsets, array<I,D> strides)
    {
        const auto index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index < size)
        {
            // compute local strides
            array<I,D> local_extents, local_strides;
            for (std::size_t i=0; i<D; ++i)  
                local_extents[i] = 1 + local_last[i] - local_first[i];
            detail::compute_strides<D>::template apply<Layout>(local_extents, local_strides);
            // compute local coordinate
            array<I,D> local_coordinate;
            detail::compute_coordinate<D>::template apply<Layout>(local_strides,local_coordinate,index);
            // add offset
            const auto memory_coordinate = local_coordinate + local_first + offsets;
            // multiply with memory strides
            const auto idx = dot(memory_coordinate, strides);
            data[idx] = buffer[index];
        }
    }
#define NCTIS 128
    template<typename Dimension, typename Layout>
    struct serialization<device::gpu, Dimension, Layout>
    {
        template<typename T, typename IndexContainer, typename Array>
        GT_FUNCTION_HOST
        static void pack(T* buffer, const IndexContainer& c, const T* m_data, const Array& m_extents, const Array& m_offsets, const Array& m_strides, void* arg)
        {
            auto stream_ptr = reinterpret_cast<cudaStream_t*>(arg);
            for (const auto& is : c)
            {
                Array local_first, local_last;
                std::copy(&is.local().first()[0], &is.local().first()[Dimension::value], local_first.data());
                std::copy(&is.local().last()[0], &is.local().last()[Dimension::value], local_last.data());
                int size = is.size();
                /*for (int blockIdx=0; blockIdx<(size+511)/512; ++blockIdx)
                    for (int threadIdx=0; threadIdx<512;++threadIdx)
                        pack_kernel_emulate<Layout>(blockIdx,512,threadIdx,size, m_data, buffer, local_first, local_last, m_extents, m_offsets, m_strides);
                */
                pack_kernel<Layout><<<(size+NCTIS)/NCTIS,NCTIS,0,*stream_ptr>>>(size, m_data, buffer, local_first, local_last, m_extents, m_offsets, m_strides);
                buffer += size;
            }
        }

        template<typename T, typename IndexContainer, typename Array>
        GT_FUNCTION_HOST
        static void unpack(const T* buffer, const IndexContainer& c, T* m_data, const Array& m_extents, const Array& m_offsets, const Array& m_strides, void* arg)
        {
            auto stream_ptr = reinterpret_cast<cudaStream_t*>(arg);
            for (const auto& is : c)
            {
                Array local_first, local_last;
                std::copy(&is.local().first()[0], &is.local().first()[Dimension::value], local_first.data());
                std::copy(&is.local().last()[0], &is.local().last()[Dimension::value], local_last.data());
                int size = is.size();

                unpack_kernel<Layout><<<(size+NCTIS)/NCTIS,NCTIS,0,*stream_ptr>>>(size, m_data, buffer, local_first, local_last, m_extents, m_offsets, m_strides);
                buffer += size;
            }
        }
    };
#endif

    // forward declaration
    template<typename T, typename Device, typename DomainDescriptor, int... Order>
    class simple_field_wrapper;

    /** @brief wraps a contiguous N-dimensional array and implements the field descriptor concept
     * @tparam T field value type
     * @tparam Device device type the data lives on
     * @tparam DomainIdType domain id type
     * @tparam Dimension N
     * @tparam Order permutation of the set {0,...,N-1} indicating storage layout (N-1 -> stride=1)*/
    template<typename T, typename Device, typename DomainIdType, int Dimension, int... Order>
    class simple_field_wrapper<T,Device,structured_domain_descriptor<DomainIdType,Dimension>, Order...>
    {
    public: // member types
        using value_type             = T;
        using device_type            = Device;
        using device_id_type         = typename device_type::id_type;
        using domain_descriptor_type = structured_domain_descriptor<DomainIdType,Dimension>;
        using dimension              = typename domain_descriptor_type::dimension;
        using layout_map             = gridtools::layout_map<Order...>;
        using domain_id_type         = DomainIdType;
        //using coordinate_type        = typename domain_descriptor_type::halo_generator_type::coordinate_type;
        using coordinate_type        = array<typename domain_descriptor_type::halo_generator_type::coordinate_type::element_type, dimension::value>;

    private: // members
        domain_id_type  m_dom_id;
        value_type*     m_data;
        coordinate_type m_strides;
        coordinate_type m_offsets;
        coordinate_type m_extents;
        device_id_type  m_device_id;

    public: // ctors
        /** @brief construcor 
         * @tparam Array coordinate-like type
         * @param dom_id local domain id
         * @param data pointer to data
         * @param offsets coordinate of first physical coordinate (not buffer) from the orign of the wrapped N-dimensional array
         * @param extents extent of the wrapped N-dimensional array (including buffer regions)*/
        template<typename Array>
        GT_FUNCTION_HOST
        simple_field_wrapper(domain_id_type dom_id, value_type* data, const Array& offsets, const Array& extents, device_id_type d_id = 0)
        : m_dom_id(dom_id), m_data(data), /*m_strides(1),*/ m_device_id(d_id)
        { 
            std::copy(offsets.begin(), offsets.end(), m_offsets.begin());
            std::copy(extents.begin(), extents.end(), m_extents.begin());
            // compute strides
            detail::compute_strides<dimension::value>::template apply<layout_map>(m_extents,m_strides);
        }
        GT_FUNCTION_HOST
        simple_field_wrapper(simple_field_wrapper&&) noexcept = default;
        GT_FUNCTION_HOST
        simple_field_wrapper(const simple_field_wrapper&) noexcept = default;

    public: // member functions
        GT_FUNCTION
        typename device_type::id_type device_id() const { return m_device_id; }
        GT_FUNCTION
        domain_id_type domain_id() const { return m_dom_id; }

        GT_FUNCTION
        const coordinate_type& extents() const noexcept { return m_extents; }
        GT_FUNCTION
        const coordinate_type& offsets() const noexcept { return m_offsets; }

        GT_FUNCTION
        value_type* data() const { return m_data; }

        /** @brief access operator
         * @param x coordinate vector with respect to offset specified in constructor
         * @return reference to value */
        GT_FUNCTION
        value_type& operator()(const coordinate_type& x) { return m_data[dot(x,m_strides)]; }
        GT_FUNCTION
        const value_type& operator()(const coordinate_type& x) const { return m_data[dot(x,m_strides)]; }

        /** @brief access operator
         * @param is coordinates with respect to offset specified in constructor
         * @return reference to value */
        template<typename... Is>
        GT_FUNCTION
        value_type& operator()(Is&&... is) { return m_data[dot(coordinate_type{is...}+m_offsets,m_strides)]; }
        template<typename... Is>
        GT_FUNCTION
        const value_type& operator()(Is&&... is) const { return m_data[dot(coordinate_type{is...}+m_offsets,m_strides)]; }

        template<typename IndexContainer>
        void pack(T* buffer, const IndexContainer& c, void* arg)
        {
            /*for (const auto& is : c)
            {
                detail::for_loop_pointer_arithmetic<dimension::value,dimension::value,layout_map>::apply(
                    [this,buffer](auto o_data, auto o_buffer)
                    {
                        buffer[o_buffer] = m_data[o_data]; 
                    }, 
                    is.local().first(), 
                    is.local().last(),
                    m_extents,
                    m_offsets
                    );
                buffer += is.size();
            }*/
            serialization<Device,dimension,layout_map>::pack(buffer, c, m_data, m_extents, m_offsets, m_strides, arg);
        }

        template<typename IndexContainer>
        void unpack(const T* buffer, const IndexContainer& c, void* arg)
        {
            /*for (const auto& is : c)
            {
                detail::for_loop_pointer_arithmetic<dimension::value,dimension::value,layout_map>::apply(
                    [this,buffer](auto o_data, auto o_buffer)
                    {
                        m_data[o_data] = buffer[o_buffer];
                    }, 
                    is.local().first(), 
                    is.local().last(),
                    m_extents,
                    m_offsets
                    );
                buffer += is.size();
            }*/
            serialization<Device,dimension,layout_map>::unpack(buffer, c, m_data, m_extents, m_offsets, m_strides, arg);
        }
    };

    /** @brief wrap a N-dimensional array (field) of contiguous memory 
     * @tparam Device device type the data lives on
     * @tparam Order permutation of the set {0,...,N-1} indicating storage layout (N-1 -> stride=1)
     * @tparam DomainIdType domain id type
     * @tparam T field value type
     * @tparam Array coordinate-like type
     * @param dom_id local domain id
     * @param data pointer to data
     * @param offsets coordinate of first physical coordinate (not buffer) from the orign of the wrapped N-dimensional array
     * @param extents extent of the wrapped N-dimensional array (including buffer regions)
     * @return wrapped field*/
    template<typename Device, int... Order, typename DomainIdType, typename T, typename Array>
    simple_field_wrapper<T,Device,structured_domain_descriptor<DomainIdType,sizeof...(Order)>, Order...>
    wrap_field(DomainIdType dom_id, T* data, const Array& offsets, const Array& extents, typename Device::id_type device_id = 0)
    {
        return {dom_id, data, offsets, extents, device_id};     
    }
} // namespace gridtools

#endif /* INCLUDED_SIMPLE_FIELD_WRAPPER_HPP */

