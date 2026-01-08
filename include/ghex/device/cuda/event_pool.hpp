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
#include <ghex/device/cuda/event.hpp>
#include <ghex/util/moved_bit.hpp>
#include <cassert>
#include <memory>
#include <vector>

namespace ghex
{
namespace device
{
/**
 * @brief	Pool of cuda events.
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
    event_pool(std::size_t expected_pool_size)
    : m_events(expected_pool_size) // Initialize events now.
    , m_next_event(0)
    {
    }

    event_pool(const event_pool&) = delete;
    event_pool& operator=(const event_pool&) = delete;
    event_pool(event_pool&& other) noexcept = default;
    event_pool& operator=(event_pool&&) noexcept = default;

  public:
    /** @brief	Get the next event of a pool.
     *
     * The function returns a new event that is not in use every time
     * it is called. If the pool is exhausted new elements are created
     * on demand.
     */
    cuda_event& get_event()
    {
        assert(!m_moved);
        while (!(m_next_event < m_events.size())) { m_events.emplace_back(cuda_event()); }

        const std::size_t event_to_use = m_next_event;
        assert(!bool(m_events[event_to_use]));
        m_next_event += 1;
        return m_events[event_to_use];
    }

    /** @brief	Mark all events in the pool as unused.
	 *
	 * Essentially resets the internal counter of the pool, this means
	 * that `get_event()` will return the very first event it returned
	 * in the beginning. This allows reusing the event without destroying
	 * and recreating them. It requires however, that a user can guarantee
	 * that the events are no longer in use.
	 */
    void rewind()
    {
        if (m_moved) { throw std::runtime_error("ERROR: Can not reset a moved pool."); }
        m_next_event = 0;
    }

    /** @brief	Clear the pool by recreating all events.
	 *
	 * The function will destroy and recreate all events in the pool.
	 * This is more costly than to rewind the pool, but allows to reuse
	 * the pool without having to ensure that the events are no longer
	 * in active use.
	 */
    void clear()
    {
        if (m_moved) { throw std::runtime_error("ERROR: Can not reset a moved pool."); }

        // NOTE: If an event is still enqueued somewhere, the CUDA runtime
        //	will made sure that it is kept alive as long as it is still used.
        m_events.clear();
        m_next_event = 0;
    }
};

} // namespace device

} // namespace ghex
