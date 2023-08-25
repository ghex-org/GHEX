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

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/lexical_cast.hpp>
#include <string>

namespace ghex
{
namespace rma
{
/** @brief Universally unique identifier used for labelling shared memory regions. */
struct uuid
{
    boost::uuids::uuid m_id;
    std::string        m_name;

    struct info
    {
        char m_name[39];
    };

    uuid()
    : m_id{boost::uuids::random_generator{}()}
    , m_name{"s-" + boost::lexical_cast<std::string>(m_id)}
    {
    }

    uuid(const uuid&) = default;
    uuid(uuid&&) = default;

    const boost::uuids::uuid& id() const noexcept { return m_id; }
    const std::string&        name() const noexcept { return m_name; }

    info get_info() const noexcept
    {
        info info_;
        for (int i = 0; i < 38; ++i) info_.m_name[i] = m_name[i];
        info_.m_name[38] = '\0';
        return info_;
    }
};

} // namespace rma
} // namespace ghex
