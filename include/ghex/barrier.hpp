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

#include <oomph/barrier.hpp>
#include <ghex/context.hpp>
#include <memory>

namespace ghex
{
#if OOMPH_ENABLE_BARRIER
class barrier
{
  private:
    std::unique_ptr<oomph::barrier> m;

  public:
    barrier(context const& c, size_t n_threads = 1)
    : m{std::make_unique<oomph::barrier>(*c.m_ctxt, n_threads)}
    {
    }

    barrier(barrier&&) noexcept = default;

  public: // public member functions
    int size() const noexcept { return m->size(); }

    // rank and thread barrier
    void operator()() const { m->operator()(); }

    void rank_barrier() const { m->rank_barrier(); }

    void thread_barrier() const { m->thread_barrier(); }
};
#endif

} // namespace ghex
