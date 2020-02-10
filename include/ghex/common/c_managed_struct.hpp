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
#ifndef INCLUDED_GHEX_COMMON_C_MANAGED_STRUCT_HPP
#define INCLUDED_GHEX_COMMON_C_MANAGED_STRUCT_HPP

#include <utility>
#include "./moved_bit.hpp"

/** @brief creates a class which manages a C struct and lifetime is managed through init and destroy C functions. */
#define GHEX_C_MANAGED_STRUCT(name, c_type, init, destroy)                              \
struct name                                                                             \
{                                                                                       \
    c_type m_struct;                                                                    \
    ::gridtools::ghex::moved_bit m_moved;                                               \
                                                                                        \
    template<typename... Args>                                                          \
    name(Args&&... args) noexcept                                                       \
    {                                                                                   \
        init(&m_struct, args...);                                                       \
    }                                                                                   \
                                                                                        \
    name(const name&) = delete;                                                         \
                                                                                        \
    name& operator=(const name&) = delete;                                              \
                                                                                        \
    name(name&& other) noexcept = default;                                              \
                                                                                        \
    name& operator=(name&& other) noexcept                                              \
    {                                                                                   \
        if (!m_moved)                                                                   \
            destroy(m_struct);                                                          \
        m_struct.~c_type();                                                             \
        ::new((void*)(&m_struct)) c_type{std::move(other.m_struct)};                    \
        m_moved = std::move(other.m_moved);                                             \
        return *this;                                                                   \
    }                                                                                   \
                                                                                        \
    ~name() noexcept                                                                    \
    {                                                                                   \
        if (!m_moved)                                                                   \
            destroy(m_struct);                                                          \
    }                                                                                   \
                                                                                        \
    operator bool() const noexcept {return !(bool)m_moved;}                             \
                                                                                        \
    operator       c_type&()       noexcept {return m_struct;}                          \
    operator const c_type&() const noexcept {return m_struct;}                          \
                                                                                        \
          c_type& get()       noexcept {return m_struct;}                               \
    const c_type& get() const noexcept {return m_struct;}                               \
                                                                                        \
};

/** @brief creates a class which holds a simle C struct */
#define GHEX_C_STRUCT(name, c_type )                                                    \
struct name                                                                             \
{                                                                                       \
    c_type m_struct;                                                                    \
    ::gridtools::ghex::moved_bit m_moved;                                               \
                                                                                        \
    name() noexcept : m_moved(true) {}                                                  \
                                                                                        \
    name(const c_type& s) noexcept                                                      \
    : m_struct{s} {}                                                                    \
                                                                                        \
    name(const name&) = delete;                                                         \
                                                                                        \
    name& operator=(const name&) = delete;                                              \
                                                                                        \
    name(name&& other) noexcept = default;                                              \
                                                                                        \
    name& operator=(name&& other) noexcept                                              \
    {                                                                                   \
        m_struct.~c_type();                                                             \
        ::new((void*)(&m_struct)) c_type{std::move(other.m_struct)};                    \
        m_moved = std::move(other.m_moved);                                             \
        return *this;                                                                   \
    }                                                                                   \
                                                                                        \
    operator bool() const noexcept {return !(bool)m_moved;}                             \
                                                                                        \
    operator       c_type&()       noexcept {return m_struct;}                          \
    operator const c_type&() const noexcept {return m_struct;}                          \
                                                                                        \
          c_type& get()       noexcept {return m_struct;}                               \
    const c_type& get() const noexcept {return m_struct;}                               \
                                                                                        \
};

#endif /* INCLUDED_GHEX_COMMON_C_MANAGED_STRUCT_HPP */
