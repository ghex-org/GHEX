/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <vector>
#include <cassert>
#include <algorithm>
#include <tuple>
#include <array>
#include <cstddef>
#include <type_traits>
#include <memory>
#include <cstring>

#include <ghex/context.hpp>
#include <ghex/util/for_each.hpp>
#include <ghex/mpi/error.hpp>
#include <ghex/mpi/status.hpp>
#include <ghex/mpi/future.hpp>

namespace ghex
{
namespace mpi
{
/** @brief special mpi communicator used for setup phase */
class communicator
{
  public:
    using rank_type = int;
    using size_type = int;

  private:
    MPI_Comm  m_comm;
    rank_type m_rank;
    size_type m_size;

  public:
    communicator(context const& c)
    : m_comm{c.transport_context()->mpi_comm()}
    {
        GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(m_comm, &m_rank));
        GHEX_CHECK_MPI_RESULT(MPI_Comm_size(m_comm, &m_size));
    }
    communicator(const communicator&) = default;
    communicator& operator=(const communicator&) = default;
    communicator(communicator&&) noexcept = default;
    communicator& operator=(communicator&&) noexcept = default;

    operator MPI_Comm() const noexcept { return m_comm; }

    inline rank_type rank() const noexcept { return m_rank; }

    inline size_type size() const noexcept { return m_size; }

    template<typename T>
    void send(int dest, int tag, const T& value) const
    {
        GHEX_CHECK_MPI_RESULT(
            MPI_Send(reinterpret_cast<const void*>(&value), sizeof(T), MPI_BYTE, dest, tag, *this));
    }

    template<typename T>
    status recv(int source, int tag, T& value) const
    {
        MPI_Status s;
        GHEX_CHECK_MPI_RESULT(
            MPI_Recv(reinterpret_cast<void*>(&value), sizeof(T), MPI_BYTE, source, tag, *this, &s));
        return {s};
    }

    template<typename T>
    void send(int dest, int tag, const T* values, int n) const
    {
        GHEX_CHECK_MPI_RESULT(MPI_Send(reinterpret_cast<const void*>(values), sizeof(T) * n,
            MPI_BYTE, dest, tag, *this));
    }

    template<typename T>
    status recv(int source, int tag, T* values, int n) const
    {
        MPI_Status s;
        GHEX_CHECK_MPI_RESULT(MPI_Recv(reinterpret_cast<void*>(values), sizeof(T) * n, MPI_BYTE,
            source, tag, *this, &s));
        return {s};
    }

    template<typename T>
    future<void> isend(int dest, int tag, const T* values, int n) const
    {
        request h;
        GHEX_CHECK_MPI_RESULT(MPI_Isend(reinterpret_cast<const void*>(values), sizeof(T) * n,
            MPI_BYTE, dest, tag, *this, &h.get()));
        return {std::move(h)};
    }

    template<typename T>
    future<void> irecv(int source, int tag, T* values, int n) const
    {
        request h;
        GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(values), sizeof(T) * n, MPI_BYTE,
            source, tag, *this, &h.get()));
        return {std::move(h)};
    }

    template<typename T>
    void broadcast(T& value, int root) const
    {
        GHEX_CHECK_MPI_RESULT(MPI_Bcast(&value, sizeof(T), MPI_BYTE, root, *this));
    }

    template<typename T>
    void broadcast(T* values, int n, int root) const
    {
        GHEX_CHECK_MPI_RESULT(MPI_Bcast(values, sizeof(T) * n, MPI_BYTE, root, *this));
    }

    template<typename T>
    future<std::vector<std::vector<T>>> all_gather(const std::vector<T>& payload,
        const std::vector<int>&                                          sizes) const
    {
        std::vector<std::vector<T>> res(size());
        for (int neigh = 0; neigh < size(); ++neigh) { res[neigh].resize(sizes[neigh]); }
        std::vector<request> m_reqs;
        m_reqs.reserve((size()) * 2);
        for (int neigh = 0; neigh < size(); ++neigh)
        {
            m_reqs.push_back(request{});
            GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(res[neigh].data()),
                sizeof(T) * sizes[neigh], MPI_BYTE, neigh, 99, *this, &m_reqs.back().get()));
        }
        for (int neigh = 0; neigh < size(); ++neigh)
        {
            m_reqs.push_back(request{});
            GHEX_CHECK_MPI_RESULT(MPI_Isend(reinterpret_cast<const void*>(payload.data()),
                sizeof(T) * payload.size(), MPI_BYTE, neigh, 99, *this, &m_reqs.back().get()));
        }
        for (auto& r : m_reqs) r.wait();

        return {std::move(res), std::move(m_reqs.front())};

        /*request h;
                    std::vector<int> displs(size());
                    std::vector<int> recvcounts(size());
                    std::vector<std::vector<T>> res(size());
                    for (int i=0; i<size(); ++i)
                    {
                        res[i].resize(sizes[i]);
                        recvcounts[i] = sizes[i]*sizeof(T);
                        displs[i] = reinterpret_cast<char*>(&res[i][0]) - reinterpret_cast<char*>(&res[0][0]);
                    }
                    GHEX_CHECK_MPI_RESULT( MPI_Iallgatherv(
                        &payload[0], payload.size()*sizeof(T), MPI_BYTE,
                        &res[0][0], &recvcounts[0], &displs[0], MPI_BYTE,
                        *this, 
                        &h.m_req));
                    return {std::move(res), std::move(h)};*/
    }

    template<typename T>
    future<std::vector<T>> all_gather(const T& payload) const
    {
        std::vector<T> res(size());
        request        h;
        GHEX_CHECK_MPI_RESULT(MPI_Iallgather(&payload, sizeof(T), MPI_BYTE, &res[0], sizeof(T),
            MPI_BYTE, *this, &h.get()));
        return {std::move(res), std::move(h)};
    }

    /** @brief computes the max element of a vector<T> among all ranks */
    template<typename T>
    T max_element(const std::vector<T>& elems) const
    {
        T    local_max{*(std::max_element(elems.begin(), elems.end()))};
        auto all_max = all_gather(local_max).get();
        return *(std::max_element(all_max.begin(), all_max.end()));
    }

    /** @brief just a helper function using custom types to be used when send/recv counts can be deduced*/
    template<typename T>
    void all_to_all(const std::vector<T>& send_buf, std::vector<T>& recv_buf) const
    {
        int comm_size = this->size();
        assert(send_buf.size() % comm_size == 0);
        assert(recv_buf.size() % comm_size == 0);
        int send_count = send_buf.size() / comm_size * sizeof(T);
        int recv_count = recv_buf.size() / comm_size * sizeof(T);
        GHEX_CHECK_MPI_RESULT(
            MPI_Alltoall(reinterpret_cast<const void*>(send_buf.data()), send_count, MPI_BYTE,
                reinterpret_cast<void*>(recv_buf.data()), recv_count, MPI_BYTE, *this));
    }

    /** @brief just a wrapper using custom types*/
    template<typename T>
    void all_to_allv(const std::vector<T>& send_buf, const std::vector<int>& send_counts,
        const std::vector<int>& send_displs, std::vector<T>& recv_buf,
        const std::vector<int>& recv_counts, const std::vector<int>& recv_displs) const
    {
        int              comm_size = this->size();
        std::vector<int> send_counts_b(comm_size), send_displs_b(comm_size),
            recv_counts_b(comm_size), recv_displs_b(comm_size);
        for (auto i = 0; i < comm_size; ++i) send_counts_b[i] = send_counts[i] * sizeof(T);
        for (auto i = 0; i < comm_size; ++i) send_displs_b[i] = send_displs[i] * sizeof(T);
        for (auto i = 0; i < comm_size; ++i) recv_counts_b[i] = recv_counts[i] * sizeof(T);
        for (auto i = 0; i < comm_size; ++i) recv_displs_b[i] = recv_displs[i] * sizeof(T);
        GHEX_CHECK_MPI_RESULT(
            MPI_Alltoallv(reinterpret_cast<const void*>(send_buf.data()), &send_counts_b[0],
                &send_displs_b[0], MPI_BYTE, reinterpret_cast<void*>(recv_buf.data()),
                &recv_counts_b[0], &recv_displs_b[0], MPI_BYTE, *this));
    }

    void barrier() { MPI_Barrier(m_comm); }

    // range view over the incoming vectors
    template<typename T>
    struct const_view
    {
        const T*    _begin;
        const T*    _end;
        const T*    begin() const noexcept { return _begin; }
        const T*    end() const noexcept { return _end; }
        std::size_t size() const noexcept { return _end - _begin; }
    };

    // This algorithm is a slight adaptation of Simon Frasch's implementation here:
    // https://github.com/AdhocMan/arbor/commit/75fa66ab983598c1c7f5e064cde4a062e9d030b5#diff-a5a30d5a5b994ff68317d61e212c8d959c209b2fa4caa3f9cd2e738b9e55ef63
    template<typename F, typename... Ts>
    void distributed_for_each(F&& f, const std::vector<Ts>&... args)
    {
        static_assert(sizeof...(args) > 0u);

        // place references to argument in tuple
        auto arg_tuple = std::forward_as_tuple(args...);

        struct vec_info
        {
            std::size_t offset; // offset in bytes
            std::size_t size;   // size in bytes
        };

        using vec_info_array = std::array<vec_info, sizeof...(args)>;
        vec_info_array     info;
        unsigned long long buffer_size = 0;

        // Compute offsets in bytes for each vector when placed in common buffer
        {
            std::size_t offset = info.size() * sizeof(vec_info);
            for_each(arg_tuple,
                [&info, &buffer_size, &offset](std::size_t i, auto&& vec)
                {
                    using T = typename std::remove_reference_t<
                        std::remove_cv_t<decltype(vec)>>::value_type;
                    static_assert(std::is_trivially_copyable_v<T>);
                    static_assert(alignof(std::max_align_t) >= alignof(T));
                    static_assert(alignof(std::max_align_t) % alignof(T) == 0);

                    // make sure alignment of offset fulfills requirement
                    const auto alignment_excess = offset % alignof(T);
                    offset += alignment_excess > 0 ? alignof(T) - (alignment_excess) : 0;

                    const auto size_in_bytes = vec.size() * sizeof(T);

                    info[i].size = size_in_bytes;
                    info[i].offset = offset;

                    buffer_size = offset + size_in_bytes;
                    offset += size_in_bytes;
                });
        }

        // compute maximum buffer size between ranks, such that we only allocate once
        unsigned long long max_buffer_size = buffer_size;
        GHEX_CHECK_MPI_RESULT(MPI_Allreduce(&buffer_size, &max_buffer_size, 1,
            MPI_UNSIGNED_LONG_LONG, MPI_MAX, *this));

        // exit if all vectors on all ranks are empty
        if (max_buffer_size == info.size() * sizeof(vec_info)) return;

        // use malloc for std::max_align_t alignment
        using buffer_t = std::unique_ptr<char[], void (*)(char*)>;
        auto deleter = [](char* ptr) { std::free(ptr); };
        auto buffer = buffer_t((char*)std::malloc(max_buffer_size), deleter);
        auto recv_buffer = buffer_t((char*)std::malloc(max_buffer_size), deleter);

        // copy offset and size info to front of buffer
        std::memcpy(buffer.get(), info.data(), info.size() * sizeof(vec_info));

        // copy each vector to each location in buffer
        for_each(arg_tuple,
            [&buffer, &info](std::size_t i, auto&& vec)
            {
                using T =
                    typename std::remove_reference_t<std::remove_cv_t<decltype(vec)>>::value_type;
                std::memcpy(buffer.get() + info[i].offset, vec.data(), vec.size() * sizeof(T));
            });

        std::tuple<const_view<Ts>...> ranges;

        const auto left_rank = rank() == 0 ? size() - 1 : rank() - 1;
        const auto right_rank = rank() == size() - 1 ? 0 : rank() + 1;

        // exchange buffer in ring pattern and apply function at each step
        for (int step = 0; step < size(); ++step)
        {
            const auto& current_info = *((const vec_info_array*)buffer.get());
            const auto  current_buffer_size = current_info.back().offset + current_info.back().size;
            // always expect to recieve the max size but send actual size.
            // MPI_recv only expects a max size, not the actual size.
            // final step does not require any exchange
            future<void> rf, sf;
            if (step < size() - 1)
            {
                rf = irecv(right_rank, 0, recv_buffer.get(), max_buffer_size);
                sf = isend(left_rank, 0, buffer.get(), current_buffer_size);
            }

            // update ranges
            for_each(arg_tuple, ranges,
                [&buffer, &current_info](std::size_t i, auto&& vec, auto&& range)
                {
                    using T = typename std::remove_reference_t<
                        std::remove_cv_t<decltype(vec)>>::value_type;

                    range = const_view<T>{(const T*)(buffer.get() + current_info[i].offset),
                        (const T*)(buffer.get() + current_info[i].offset + current_info[i].size)};
                });

            // call provided function with ranges pointing to current buffer
            std::apply(f, std::tuple_cat(std::make_tuple((rank() + step) % size()), ranges));

            // final step does not require any exchange
            if (step < size() - 1)
            {
                rf.wait();
                sf.wait();
                buffer.swap(recv_buffer);
            }
        }
    }
};

} // namespace mpi
} // namespace ghex
