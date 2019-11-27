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
#ifndef INCLUDED_GHEX_STRUCTURED_SIMPLE_FIELD_WRAPPER_HPP
#define INCLUDED_GHEX_STRUCTURED_SIMPLE_FIELD_WRAPPER_HPP

#include "./field_utils.hpp"
#include "./domain_descriptor.hpp"
#include <cstring>
#include <cstdint>
#include <gridtools/common/array.hpp>
#include "../arch_traits.hpp"


#define NCTIS 128

namespace gridtools {

    template <typename T, size_t D>
    GT_FUNCTION
    array<T,D> operator+(array<T,D> a, const array<T,D>& b)
    {
        for (std::size_t i=0u; i<D; ++i) a[i] += b[i];
        return a;
    }
    template <typename T, size_t D>
    GT_FUNCTION
    array<T,D> operator-(array<T,D> a, const array<T,D>& b)
    {
        for (std::size_t i=0u; i<D; ++i) a[i] -= b[i];
        return a;
    }
    template <typename T, typename U, size_t D>
    GT_FUNCTION
    typename std::common_type<T, U>::type dot(const array<T,D>& a, const array<U,D>& b)
    {
        typename std::common_type<T, U>::type res = a[0]*b[0];
        for (std::size_t i=1u; i<D; ++i) res+=a[i]*b[i];
        return res;
    }

namespace ghex {
namespace structured {    
#ifdef __CUDACC__
    template<typename Layout, typename T, std::size_t D, typename I, typename S>
    __global__ void pack_kernel(int size, const T* data, T* buffer, 
                                array<I,D> local_first, array<I,D> local_strides,
                                array<S,D> byte_strides, array<I,D> offsets)
    {
        const auto index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index < size)
        {
            // compute local coordinate
            array<I,D> local_coordinate;
            detail::compute_coordinate<D>::template apply<Layout>(local_strides,local_coordinate,index);
            // add offset
            const auto memory_coordinate = local_coordinate + local_first + offsets;
            // multiply with memory strides
            const auto idx = dot(memory_coordinate, byte_strides);
            buffer[index] = *reinterpret_cast<const T*>((const char*)data + idx);
        }
    }

    template<typename Layout, typename T, std::size_t D, typename I, typename S>
    __global__ void unpack_kernel(int size, T* data, const T* buffer, 
                                array<I,D> local_first, array<I,D> local_strides,
                                array<S,D> byte_strides, array<I,D> offsets)
    {
        const auto index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index < size)
        {
            // compute local coordinate
            array<I,D> local_coordinate;
            detail::compute_coordinate<D>::template apply<Layout>(local_strides,local_coordinate,index);
            // add offset
            const auto memory_coordinate = local_coordinate + local_first + offsets;
            // multiply with memory strides
            const auto idx = dot(memory_coordinate, byte_strides);
            *reinterpret_cast<T*>((char*)data + idx) = buffer[index];
        }
    }
#endif

    template<typename Arch, typename Dimension, typename Layout>
    struct serialization
    {
        template<typename T, typename IndexContainer, typename Strides, typename Array>
        GT_FUNCTION_HOST
        static void pack(T* buffer, const IndexContainer& c, const T* m_data, const Strides& m_byte_strides, 
                         const Array& m_offsets, void*)
        {
            for (const auto& is : c)
            {
                ::gridtools::ghex::detail::for_loop_pointer_arithmetic<Dimension::value,Dimension::value,Layout>::apply(
                    [m_data,buffer](auto o_data, auto o_buffer)
                    {
                        *reinterpret_cast<T*>(reinterpret_cast<char*>(buffer)+o_buffer) = 
                        *reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_data)+o_data); 
                    }, 
                    is.local().first(), 
                    is.local().last(),
                    m_byte_strides,
                    m_offsets
                    );
                buffer += is.size();
            }
        }

        template<typename T, typename IndexContainer, typename Strides, typename Array>
        GT_FUNCTION_HOST
        static void unpack(const T* buffer, const IndexContainer& c, T* m_data, const Strides& m_byte_strides, 
                           const Array& m_offsets, void*)
        {
            for (const auto& is : c)
            {
                ::gridtools::ghex::detail::for_loop_pointer_arithmetic<Dimension::value,Dimension::value,Layout>::apply(
                    [m_data,buffer](auto o_data, auto o_buffer)
                    {
                        *reinterpret_cast<T*>(reinterpret_cast<char*>(m_data)+o_data) = 
                        *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer)+o_buffer); 
                    }, 
                    is.local().first(), 
                    is.local().last(),
                    m_byte_strides,
                    m_offsets
                    );
                buffer += is.size();
            }
        }
    };


#ifdef __CUDACC__
    template<typename Dimension, typename Layout>
    struct serialization<gpu, Dimension, Layout>
    {
        template<typename T, typename IndexContainer, typename Strides, typename Array>
        GT_FUNCTION_HOST
        static void pack(T* buffer, const IndexContainer& c, const T* m_data, const Strides& m_byte_strides, 
                         const Array& m_offsets, void* arg)
        {
            auto stream_ptr = reinterpret_cast<cudaStream_t*>(arg);
            for (const auto& is : c)
            {
                Array local_first, local_last;
                std::copy(&is.local().first()[0], &is.local().first()[Dimension::value], local_first.data());
                std::copy(&is.local().last()[0], &is.local().last()[Dimension::value], local_last.data());
                Array local_extents, local_strides;
                for (std::size_t i=0; i<Dimension::value; ++i)  
                    local_extents[i] = 1 + local_last[i] - local_first[i];
                detail::compute_strides<Dimension::value>::template apply<Layout>(local_extents, local_strides);
                const int size = is.size();
                pack_kernel<Layout><<<(size+NCTIS-1)/NCTIS,NCTIS,0,*stream_ptr>>>(
                    size, m_data, buffer, local_first, local_strides, m_byte_strides, m_offsets);
                buffer += size;
            }
        }

        template<typename T, typename IndexContainer, typename Strides, typename Array>
        GT_FUNCTION_HOST
        static void unpack(const T* buffer, const IndexContainer& c, T* m_data, const Strides& m_byte_strides, 
                           const Array& m_offsets, void* arg)
        {
            auto stream_ptr = reinterpret_cast<cudaStream_t*>(arg);
            for (const auto& is : c)
            {
                Array local_first, local_last;
                std::copy(&is.local().first()[0], &is.local().first()[Dimension::value], local_first.data());
                std::copy(&is.local().last()[0], &is.local().last()[Dimension::value], local_last.data());
                Array local_extents, local_strides;
                for (std::size_t i=0; i<Dimension::value; ++i)  
                    local_extents[i] = 1 + local_last[i] - local_first[i];
                detail::compute_strides<Dimension::value>::template apply<Layout>(local_extents, local_strides);
                const int size = is.size();
                unpack_kernel<Layout><<<(size+NCTIS-1)/NCTIS,NCTIS,0,*stream_ptr>>>(
                        size, m_data, buffer, local_first, local_strides, m_byte_strides, m_offsets);
                buffer += size;
            }
        }
    };
#endif

    struct padding_256 {};

    /** @brief wraps a contiguous N-dimensional array and implements the field descriptor concept
     * @tparam T field value type
     * @tparam Arch device type the data lives on
     * @tparam DomainDescriptor domain type
     * @tparam Order permutation of the set {0,...,N-1} indicating storage layout (N-1 -> stride=1)*/
    template<typename T, typename Arch, typename DomainDescriptor, int... Order>
    class simple_field_wrapper
    {
    public: // member types
        using value_type             = T;
        using arch_type              = Arch;
        using device_id_type         = typename arch_traits<arch_type>::device_id_type;
        using domain_descriptor_type = DomainDescriptor;
        using dimension              = typename domain_descriptor_type::dimension;
        using layout_map             = ::gridtools::layout_map<Order...>;
        using domain_id_type         = typename DomainDescriptor::domain_id_type;
        using coordinate_type        = ::gridtools::array<typename domain_descriptor_type::coordinate_type::value_type, dimension::value>;
        using strides_type           = ::gridtools::array<std::size_t, dimension::value>;

    private: // members
        domain_id_type  m_dom_id;
        value_type*     m_data;
        coordinate_type m_offsets;
        coordinate_type m_extents;
        device_id_type  m_device_id;
        strides_type    m_byte_strides;

    public: // ctors
        
        simple_field_wrapper() noexcept = default;

        /** @brief construcor 
         * @tparam Array coordinate-like type
         * @param dom_id local domain id
         * @param data pointer to data
         * @param offsets coordinate of first physical coordinate (not buffer) from the orign of the wrapped N-dimensional array
         * @param extents extent of the wrapped N-dimensional array (including buffer regions)*/
        template<typename Array>
        GT_FUNCTION_HOST
        simple_field_wrapper(domain_id_type dom_id, value_type* data, const Array& offsets, const Array& extents, device_id_type d_id = 0)
        : m_dom_id(dom_id), m_data(data), m_device_id(d_id)
        { 
            std::copy(offsets.begin(), offsets.end(), m_offsets.begin());
            std::copy(extents.begin(), extents.end(), m_extents.begin());
            // compute strides
            detail::compute_strides<dimension::value>::template apply<layout_map,value_type>(m_extents,m_byte_strides,0u);
        }

        template<typename Array0, typename Array1, typename Array2>
        GT_FUNCTION_HOST
        simple_field_wrapper(domain_id_type dom_id, value_type* data, const Array0& offsets, const Array1& extents, const Array2& byte_strides, device_id_type d_id = 0)
        : m_dom_id(dom_id), m_data(data), m_device_id(d_id)
        { 
            std::copy(offsets.begin(), offsets.end(), m_offsets.begin());
            std::copy(extents.begin(), extents.end(), m_extents.begin());
            std::copy(byte_strides.begin(), byte_strides.end(), m_byte_strides.begin());
        }

        template<typename Array>
        GT_FUNCTION_HOST
        simple_field_wrapper(domain_id_type dom_id, value_type* data, const Array& offsets, const Array& extents, padding_256, device_id_type d_id = 0)
        : m_dom_id(dom_id), m_data(data), m_device_id(d_id)
        { 
            std::copy(offsets.begin(), offsets.end(), m_offsets.begin());
            std::copy(extents.begin(), extents.end(), m_extents.begin());
            // pad stride 1 dimension
            const auto ext_1 = m_extents[layout_map::template find<dimension::value-1>()]*sizeof(value_type);
            const std::size_t padding = ((ext_1+255)/256)*256 - ext_1;
            // compute strides
            detail::compute_strides<dimension::value>::template apply<layout_map,value_type>(m_extents,m_byte_strides,padding);
            // compute offset in bytes of the first physical point
            auto offset_origin = dot(m_offsets, m_byte_strides);
            // compute pointer value of first physical point 
            auto ptr_origin = reinterpret_cast<std::uintptr_t>((void*)data) + offset_origin; 
            // compute delta
            const auto delta = ((ptr_origin + 255)/256) * 256 - ptr_origin;
            // adjust base pointer
            m_data = reinterpret_cast<value_type*>(reinterpret_cast<char*>(m_data)+delta);
        }
        //GT_FUNCTION_HOST
        simple_field_wrapper(simple_field_wrapper&&) noexcept = default;
        //GT_FUNCTION_HOST
        simple_field_wrapper(const simple_field_wrapper&) noexcept = default;

        //GT_FUNCTION_HOST
        simple_field_wrapper& operator=(simple_field_wrapper&&) noexcept = default;
        //GT_FUNCTION_HOST
        simple_field_wrapper& operator=(const simple_field_wrapper&) noexcept = default;

    public: // member functions
        GT_FUNCTION
        typename arch_traits<arch_type>::device_id_type device_id() const { return m_device_id; }
        GT_FUNCTION
        domain_id_type domain_id() const { return m_dom_id; }

        GT_FUNCTION
        const coordinate_type& extents() const noexcept { return m_extents; }
        GT_FUNCTION
        const coordinate_type& offsets() const noexcept { return m_offsets; }
        GT_FUNCTION
        const strides_type& byte_strides() const noexcept { return m_byte_strides; }

        GT_FUNCTION
        value_type* data() const { return m_data; }

        GT_FUNCTION
        void set_data(value_type* ptr) { m_data = ptr; }

        /** @brief access operator
         * @param x coordinate vector with respect to offset specified in constructor
         * @return reference to value */
        GT_FUNCTION
        value_type& operator()(const coordinate_type& x) { return *reinterpret_cast<T*>((char*)m_data +dot(x,m_byte_strides)); }
        GT_FUNCTION
        const value_type& operator()(const coordinate_type& x) const { return *reinterpret_cast<const T*>((const char*)m_data +dot(x,m_byte_strides)); }

        /** @brief access operator
         * @param is coordinates with respect to offset specified in constructor
         * @return reference to value */
        template<typename... Is>
        GT_FUNCTION
        value_type& operator()(Is&&... is) { return *reinterpret_cast<T*>((char*)m_data+dot(coordinate_type{is...}+m_offsets,m_byte_strides)); }
        template<typename... Is>
        GT_FUNCTION
        const value_type& operator()(Is&&... is) const { return *reinterpret_cast<const T*>((const char*)m_data+dot(coordinate_type{is...}+m_offsets,m_byte_strides)); }

        template<typename IndexContainer>
        void pack(T* buffer, const IndexContainer& c, void* arg)
        {
            serialization<Arch,dimension,layout_map>::pack(buffer, c, m_data, m_byte_strides, m_offsets, arg);
        }

        template<typename IndexContainer>
        void unpack(const T* buffer, const IndexContainer& c, void* arg)
        {
            serialization<Arch,dimension,layout_map>::unpack(buffer, c, m_data, m_byte_strides, m_offsets, arg);
        }
    };
} // namespace structured

    /** @brief wrap a N-dimensional array (field) of contiguous memory 
     * @tparam Arch device type the data lives on
     * @tparam Order permutation of the set {0,...,N-1} indicating storage layout (N-1 -> stride=1)
     * @tparam DomainIdType domain id type
     * @tparam T field value type
     * @tparam Array coordinate-like type
     * @param dom_id local domain id
     * @param data pointer to data
     * @param offsets coordinate of first physical coordinate (not buffer) from the orign of the wrapped N-dimensional array
     * @param extents extent of the wrapped N-dimensional array (including buffer regions)
     * @return wrapped field*/
    template<typename Arch, int... Order, typename DomainIdType, typename T, typename Array>
    structured::simple_field_wrapper<T,Arch,structured::domain_descriptor<DomainIdType,sizeof...(Order)>, Order...>
    wrap_field(DomainIdType dom_id, T* data, const Array& offsets, const Array& extents, typename arch_traits<Arch>::device_id_type device_id = 0)
    {
        return {dom_id, data, offsets, extents, device_id};     
    }
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_SIMPLE_FIELD_WRAPPER_HPP */

