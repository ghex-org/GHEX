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

#include <ghex/config.hpp>
#include <ghex/device/cuda/error.hpp>
#include <ghex/device/cuda/runtime.hpp>
#include <ghex/util/moved_bit.hpp>
#include <assert.h>
#include <memory>
#include <vector>

namespace ghex
{
namespace device
{
struct cuda_event
{
    cudaEvent_t           m_event;
    ghex::util::moved_bit m_moved;

    cuda_event(){GHEX_CHECK_CUDA_RESULT(cudaEventCreateWithFlags(&m_event,
        cudaEventDisableTiming))} cuda_event(const cuda_event&) = delete;
    cuda_event& operator=(const cuda_event&) = delete;
    cuda_event(cuda_event&& other) = default;
    cuda_event& operator=(cuda_event&&) = default;

    ~cuda_event()
    {
        if (!m_moved) { GHEX_CHECK_CUDA_RESULT_NO_THROW(cudaEventDestroy(m_event)) }
    }

    operator bool() const noexcept { return m_moved; }
    operator cudaEvent_t() const noexcept { return m_event; }
    cudaEvent_t&       get() noexcept { return m_event; }
    const cudaEvent_t& get() const noexcept { return m_event; }
};

/** @brief thin wrapper around a cuda stream */
struct stream
{
    cudaStream_t          m_stream;
    ghex::util::moved_bit m_moved;

    stream(){GHEX_CHECK_CUDA_RESULT(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking))}

    stream(const stream&) = delete;
    stream& operator=(const stream&) = delete;
    stream(stream&& other) = default;
    stream& operator=(stream&&) = default;

    ~stream()
    {
        if (!m_moved) { GHEX_CHECK_CUDA_RESULT_NO_THROW(cudaStreamDestroy(m_stream)) }
    }

    operator bool() const noexcept { return m_moved; }

    operator cudaStream_t() const noexcept { return m_stream; }

    cudaStream_t&       get() noexcept { return m_stream; }
    const cudaStream_t& get() const noexcept { return m_stream; }

    void sync()
    {
        // busy wait here
        GHEX_CHECK_CUDA_RESULT(cudaStreamSynchronize(m_stream))
    }
};

/**
 * @breif	Pool of cuda events.
 *
 * Essentially a pool of events that can be used and reused one by one.
 * The main function is `get_event()` which returns an unused event.
 * To reuse an event the pool can either be rewinded, i.e. start again
 * with the first event, which requires that the user guarantees that
 * all events are no longer in use. The second way is to reset the pool
 * i.e. to destroy and recreate all events, which is much more expensive.
 *
 * Note that the pool is not thread safe.
 *
 * Todo:
 * - Maybe create a compile time size.
 * - Speed up `reset_pool()` by limiting recreation.
 */
struct event_pool
{
  private: // members
    std::vector<cuda_event> m_events;
    std::size_t             m_next_event;
    ghex::util::moved_bit   m_moved;

  public: // constructors
    event_pool(std::size_t pool_size)
    : m_events(pool_size)
    , m_next_event(0)
    {
        if (pool_size == 0) { throw std::invalid_argument("ERROR: Pool size can not be zero."); };
    };

    event_pool(const event_pool&) = delete;
    event_pool& operator=(const event_pool&) = delete;
    event_pool(event_pool&& other) = default;
    event_pool& operator=(event_pool&&) = default;

  public:
    /** @brief	Get the next event of a pool.
	 *
	 * The function returns the next not yet used event.
	 * If the pool is exhausted the behaviour depends on `wrap_around`.
	 * If it is `true`, then the event that was returned at the beginning
	 * is returned again. Because this might be dangerous, the default
	 * behaviour is to generate an error.
	 */
    cuda_event& get_event(bool wrap_around)
    {
        if (m_next_event <= m_events.size())
        {
            if (m_moved) { throw std::runtime_error("ERROR: pool has been moved."); };
            if (wrap_around) { m_next_event = 0; }
            else { throw std::runtime_error("ERROR: Exhausted event pool"); };
        };

        const std::size_t event_to_use = m_next_event;
        m_next_event += 1;
        assert(m_next_event < m_events.size());
        return m_events[event_to_use];
    };

    cuda_event& get_event() { return get_event(false); };

    /** @brief	Mark all events in the pool as unused.
	 *
	 * Essentially resets the internal counter of the pool, this means
	 * that `get_event()` will return the very first event it returned
	 * in the beginning. This allows reusing the event without destroying
	 * and recreating them. It requires however, that a user can guarantee
	 * that the events are no longer in use.
	 */
    void rewind_pool()
    {
        assert(!m_moved);
        m_next_event = 0;
    };

    /** @brief	Resets the pool by recreating all events.
	 *
	 * The function will destroy and recreate all events in the pool.
	 * This is more costly than to rewind the pool, but allows to reuse
	 * the pool without having to ensure that the events are no longer
	 * in active use.
	 */
    void reset_pool()
    {
        if (m_moved) { throw std::runtime_error("ERROR: Can not reset a moved pool."); };

        const auto old_size = m_events.size();
        //NOTE: If an event is still enqueued somewhere, the CUDA runtime
        //	will made sure that it is kept alive as long as it is still used.
        //NOTE: Without wrap around we could just recreate the events that have
        //	been used, but without knowing it, we must recreate all.
        m_events.clear();
        m_events.resize(old_size);
        m_next_event = 0;
    };
};

} // namespace device

} // namespace ghex
