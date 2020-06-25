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
#ifndef INCLUDED_GHEX_STRUCTURED_REMOTE_THREAD_RANGE_HPP
#define INCLUDED_GHEX_STRUCTURED_REMOTE_THREAD_RANGE_HPP

#include <cstring>
#include <vector>
#include <gridtools/common/host_device.hpp>

#include "../remote_range_traits.hpp"
#include "../transport_layer/ri/types.hpp"
#include "../transport_layer/ri/thread/access_guard.hpp"

#include <iostream>

namespace gridtools {
namespace ghex {
namespace structured {

template<typename Range>
struct range_iterator
{
    using coordinate = typename Range::coordinate;
    using chunk = tl::ri::chunk;
    using size_type = tl::ri::size_type;

    Range*      m_range;
    size_type   m_index;
    coordinate  m_coord;

    range_iterator(Range* r, size_type idx, const coordinate& coord)
    : m_range{r}
    , m_index{idx}
    , m_coord{coord}
    {}
    range_iterator(const range_iterator&) = default;
    range_iterator(range_iterator&&) = default;

    chunk     operator*() const noexcept { return m_range->get_chunk(m_coord); }
    void      operator++() noexcept { m_index = m_range->inc(m_index, m_coord); }
    void      operator--() noexcept { m_index = m_range->inc(m_index, -1, m_coord); }
    void      operator+=(size_type n) noexcept { m_index = m_range->inc(m_index, n, m_coord); }
    size_type sub(const range_iterator& other) const { return m_index - other.m_index; }
    bool      equal(const range_iterator& other) const { return m_index == other.m_index; }
    bool      lt(const range_iterator& other) const { return m_index < other.m_index; }
};

template<typename Field, typename Enable = void> // enable for gpu
struct field_view
{
    using layout = typename Field::layout_map;
    using dimension = typename Field::dimension;
    using value_type = typename Field::value_type;
    using coordinate = typename Field::coordinate_type;
    using strides_type = typename Field::strides_type;
    using guard_type = tl::ri::thread::access_guard;
    using guard_view_type = tl::ri::thread::access_guard_view;
    using size_type = tl::ri::size_type;

    Field* m_field;
    coordinate m_offset;
    coordinate m_extent;
    coordinate m_begin;
    coordinate m_end;
    coordinate m_reduced_stride;
    size_type m_size;

    void print_info()
    {
        std::cout << "view on field " << m_field << " with start at "
            << m_offset[0] << ", "
            << m_offset[1] << ", "
            << m_offset[2] << " and size " 
            << m_extent[0] << ", "
            << m_extent[1] << ", "
            << m_extent[2] << std::endl;
    }
    
    template<typename Array>
    field_view(Field& f, const Array& offset, const Array& extent)
    : m_field(&f)
    {
        static constexpr auto I = layout::template find<dimension::value-1>();
        m_size = 1;
        for (unsigned int i=0; i<dimension::value; ++i)
        {
            m_offset[i] = offset[i];
            m_extent[i] = extent[i];
            m_begin[i] = 0;
            m_end[i] = extent[i]-1;
            m_reduced_stride[i] = m_field->byte_strides()[i] / m_field->extents()[I];
            m_size *= extent[i];
        }
        m_end[I] = extent[I];
        m_size /= extent[layout::template find<dimension::value-1>()];
    }

    field_view(const field_view&) = default;
    field_view(field_view&&) = default;
    
    GT_FUNCTION
    value_type& operator()(const coordinate& x) {
        return m_field->operator()(x+m_offset);
    }
    GT_FUNCTION
    const value_type& operator()(const coordinate& x) const {
        return m_field->operator()(x+m_offset);
    }
};

namespace detail {
template<unsigned int Dim, unsigned int D, typename Layout>
struct inc_coord
{
    template<typename Coord>
    static void fn(Coord& coord, const Coord& ext)
    {
        static constexpr auto I = Layout::template find<Dim-D-1>();
        if (coord[I] == ext[I] - 1)
        {
            coord[I] = 0;
            inc_coord<Dim, D + 1, Layout>::fn(coord, ext);
        }
        else
            coord[I] += 1;
    }
};
template<unsigned int Dim, typename Layout>
struct inc_coord<Dim, Dim, Layout>
{
    template<typename Coord>
    static void fn(Coord& coord, const Coord& ext)
    {
        static constexpr auto I = Layout::template find<Dim-1>();
        for (unsigned int i = 0; i < Dim; ++i) coord[i] = ext[i] - 1;
        coord[I] = ext[I];
    }
};
} // namespace detail

template<typename Field, typename Enable = void> // enable for gpu
struct remote_thread_range
{
    using view_type = field_view<Field>;
    using layout = typename Field::layout_map;
    using dimension = typename Field::dimension;
    using value_type = typename Field::value_type;
    using coordinate = typename Field::coordinate_type;
    using strides_type = typename Field::strides_type;
    using guard_type = tl::ri::thread::access_guard;
    using guard_view_type = tl::ri::thread::access_guard_view;
    using size_type = tl::ri::size_type;
    using iterator = range_iterator<remote_thread_range>;

    guard_view_type   m_guard;
    view_type         m_view;
    size_type         m_chunk_size;
    
    remote_thread_range(const view_type& v, guard_type& g) noexcept
    : m_guard{g}
    , m_view{v}
    , m_chunk_size{(size_type)(m_view.m_extent[layout::template find<dimension::value-1>()] * sizeof(value_type))}
    {}
    
    remote_thread_range(const remote_thread_range&) = default;
    remote_thread_range(remote_thread_range&&) = default;

    void print_info() { m_view.print_info(); }
    iterator  begin() const { return {const_cast<remote_thread_range*>(this), 0, m_view.m_begin}; }
    iterator  end()   const { return {const_cast<remote_thread_range*>(this), m_view.m_size, m_view.m_end}; }
    size_type buffer_size() const { return m_chunk_size; }

    // these functions are called at the remote site upon deserializing and reconstructing the range
    // and can be used to allocate state
    void init(tl::ri::remote_host_)   {}
    void init(tl::ri::remote_device_) {}
    void exit(tl::ri::remote_host_)   {}
    void exit(tl::ri::remote_device_) {}
    
    //static void put(tl::ri::chunk c, const tl::ri::byte* ptr, tl::ri::remote_host_) {
    //    // host to host put
    //    std::memcpy(c.data(), ptr, c.size());
    //}
    //static void put(tl::ri::chunk, const tl::ri::byte*, tl::ri::remote_device_) {}

    static iterator put(iterator it, const tl::ri::byte* ptr, tl::ri::remote_host_) {
    //    // de-virtualize iterator
    //    //iterator it_ = dynamic_cast<iterator&>(it);
        tl::ri::chunk c = *it; 
        // host to host put
        std::memcpy(c.data(), ptr, c.size());
        return it;
    }
    //template<typename IteratorBase>
    //static void put(iterator& it, const tl::ri::byte* ptr, tl::ri::remote_device_) { }


    void start_local_epoch() { m_guard.start_local_epoch(); }
    void end_local_epoch()   { m_guard.end_local_epoch(); }

    void start_remote_epoch(tl::ri::remote_host_)   { m_guard.start_remote_epoch(); }
    void end_remote_epoch(tl::ri::remote_host_)     { m_guard.end_remote_epoch(); }
    void start_remote_epoch(tl::ri::remote_device_) { m_guard.start_remote_epoch(); }
    void end_remote_epoch(tl::ri::remote_device_)   { m_guard.end_remote_epoch(); }
    
    tl::ri::chunk get_chunk(const coordinate& coord) const noexcept {
        auto ptr = const_cast<tl::ri::byte*>(reinterpret_cast<const tl::ri::byte*>(&(m_view(coord))));
        return {ptr, m_chunk_size};
    }

    size_type inc(size_type index, size_type n, coordinate& coord) const noexcept {
        if (n < 0 && -n > index)
        {
            coord = m_view.m_begin;
            return 0;
        }
        index += n;
        if (index >= m_view.m_size)
        {
            coord = m_view.m_end;
            return m_view.m_size;
        }
        else
        {
            auto idx = index;
            static constexpr auto I = layout::template find<dimension::value-1>();
            coord[I] = 0;
            for (unsigned int d = 0; d < dimension::value; ++d)
            {
                const auto i = layout::find(d);
                coord[i] = index / m_view.m_reduced_stride[i];
                index -= coord[i] * m_view.m_reduced_stride[i];
            }
            return idx;
        }
    }

    size_type inc(size_type index, coordinate& coord) const noexcept {
        static constexpr auto I = layout::template find<dimension::value-1>();
        if (index + 1 >= m_view.m_size)
        {
            coord = m_view.m_end;
            return m_view.m_size;
        }
        else
        {
            coord[I] = 0;
            detail::inc_coord<dimension::value, 1, layout>::fn(coord, m_view.m_extent);
            return index + 1;
        }
    }
};

static std::mutex mtx;
template<typename Field>
struct remote_thread_range_generator
{
    using range_type = remote_thread_range<Field>;

    template<typename RangeFactory, typename Communicator>
    struct target_range
    {
        Communicator m_comm;
        tl::ri::thread::access_guard m_guard;
        field_view<Field> m_view;
        typename RangeFactory::range_type m_local_range;
        int m_tag;
        std::vector<tl::ri::byte> m_archive;

        template<typename Coord>
        target_range(const Communicator& comm, Field& f, const Coord& first, const Coord& last, int tag)
        : m_comm{comm}
        , m_guard{}
        , m_view{f, first, last-first+1}
        , m_local_range{RangeFactory::template create<range_type>(m_view,m_guard)}
        , m_tag{tag}
        {
            m_archive.resize(RangeFactory::serial_size);
            RangeFactory::serialize(m_local_range, m_archive.data());
        }

        void send()
        {
            m_guard.init(tl::ri::thread::access_guard::remote);
            { std::lock_guard<std::mutex> lock(mtx);
            std::cout << "sending with tag " << m_tag << " from thread " << m_comm.thread_id() << std::endl;
            }
            m_comm.send(m_archive, m_comm.rank(), m_tag).wait();
        }
    };

    template<typename RangeFactory, typename Communicator>
    struct source_range
    {
        Communicator m_comm;
        field_view<Field> m_view;
        typename RangeFactory::range_type m_remote_range;
        int m_tag;
        typename Communicator::template future<void> m_request;
        std::vector<tl::ri::byte> m_buffer;
        typename RangeFactory::range_type::iterator_type m_pos;

        template<typename Coord>
        source_range(const Communicator& comm, Field& f, const Coord& first, const Coord& last, int tag)
        : m_comm{comm}
        , m_view{f, first, last-first+1}
        , m_tag{tag}
        {
            m_buffer.resize(RangeFactory::serial_size);
            { std::lock_guard<std::mutex> lock(mtx);
            std::cout << "posting recv  with tag " << m_tag << " from thread " << m_comm.thread_id() << std::endl;
            }
            m_request = m_comm.recv(m_buffer, m_comm.rank(), m_tag);
        }

        void recv()
        {
            m_request.wait();
            m_remote_range = RangeFactory::deserialize(tl::ri::host, m_buffer.data());
            { std::lock_guard<std::mutex> lock(mtx);
                std::cout << "deserialized range:\n";
                m_remote_range.print_info();
            }
            m_buffer.resize(m_remote_range.buffer_size());
            m_pos = m_remote_range.begin();
        }
    };
};

} // namespace structured

template<>
struct remote_range_traits<structured::remote_thread_range_generator>
{
    template<typename Communicator>
    static bool is_local(Communicator comm, int remote_rank)
    {
        return comm.rank() == remote_rank;
    }
};

} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_REMOTE_THREAD_RANGE_HPP */
