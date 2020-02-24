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
#ifndef INCLUDED_GHEX_COMMON_COORDINATE_HPP
#define INCLUDED_GHEX_COMMON_COORDINATE_HPP

#include <array>
#include <algorithm>
#include <iosfwd>

namespace gridtools {

    namespace ghex {

        template<typename Array>
        struct coordinate
        {
        public: // member types
            using array_type     = Array;
            using iterator       = typename array_type::iterator;
            using const_iterator = typename array_type::const_iterator;
            using dimension      = std::tuple_size<array_type>;
            using element_type   = typename array_type::value_type;
            using value_type     = element_type;

        public: // static members
            static constexpr int size() noexcept { return dimension::value; }

        private: // members
            array_type m_coord;

        public: // print
            template< class CharT, class Traits>
            friend std::basic_ostream<CharT,Traits>& operator<<(std::basic_ostream<CharT,Traits>& os, const coordinate& c)
            {
                os << "{";
                for (int i=0; i<size()-1; ++i) os << c.m_coord[i] << ", ";
                os << c.m_coord[size()-1] << "}";
                return os;
            }

        public: // ctors
            coordinate() noexcept = default;
            coordinate(const coordinate&) noexcept = default;
            coordinate(coordinate&&) noexcept = default;
            coordinate(const array_type& a) noexcept : m_coord(a) {}
            coordinate(array_type&& a) noexcept : m_coord(std::move(a)) {}
            template<typename I0, typename I1, typename... Is>
            coordinate(I0&& i0, I1&& i1, Is&&... components) noexcept : m_coord{i0,i1,components...} {}
            template<typename I>
            coordinate(I scalar) noexcept
            {
                for (auto& x : m_coord) x = scalar;
            }

        public: // assignment
            coordinate& operator=(const coordinate&) noexcept = default;
            coordinate& operator=(coordinate&&) noexcept = default;

        public: // comparison
            bool operator==(const coordinate& other) const noexcept
            {
                for (int i=0; i<size(); ++i)
                    if (m_coord[i] != other.m_coord[i])
                        return false;
                return true;
            }
            bool operator!=(const coordinate& other) const noexcept
            {
                return !(*this == other);
            }
            bool operator<(const coordinate& other) const noexcept
            {
                for (int i=0; i<size(); ++i)
                    if (m_coord[i] >= other.m_coord[i])
                        return false;
                return true;
            }
            bool operator>(const coordinate& other) const noexcept
            {
                for (int i=0; i<size(); ++i)
                    if (m_coord[i] <= other.m_coord[i])
                        return false;
                return true;
            }
            bool operator<=(const coordinate& other) const noexcept
            {
                for (int i=0; i<size(); ++i)
                    if (m_coord[i] > other.m_coord[i])
                        return false;
                return true;
            }
            bool operator>=(const coordinate& other) const noexcept
            {
                for (int i=0; i<size(); ++i)
                    if (m_coord[i] < other.m_coord[i])
                        return false;
                return true;
            }

        public: // implicit conversion
            operator array_type() const noexcept { return m_coord; }

        public: // access
            const auto& operator[](int i) const noexcept { return m_coord[i]; }
            auto& operator[](int i) noexcept { return m_coord[i]; }

        public: // iterators
            iterator begin() noexcept { return m_coord.begin(); }
            const_iterator begin() const noexcept { return m_coord.cbegin(); }
            const_iterator cbegin() const noexcept { return m_coord.cbegin(); }

            iterator end() noexcept { return m_coord.end(); }
            const_iterator end() const noexcept { return m_coord.cend(); }
            const_iterator cend() const noexcept { return m_coord.cend(); }

        public: // arithmentic operators
            coordinate& operator+=(const coordinate& c) noexcept
            {
                for (int i=0; i<size(); ++i) m_coord[i] += c.m_coord[i];
                return *this;
            }
            template<typename I>
            coordinate& operator+=(I scalar) noexcept
            {
                for (int i=0; i<size(); ++i) m_coord[i] += scalar;
                return *this;
            }
            coordinate& operator-=(const coordinate& c) noexcept
            {
                for (int i=0; i<size(); ++i) m_coord[i] -= c.m_coord[i];
                return *this;
            }
            template<typename I>
            coordinate& operator-=(I scalar) noexcept
            {
                for (int i=0; i<size(); ++i) m_coord[i] -= scalar;
                return *this;
            }
        };

        template<typename T>
        using is_coordinate = std::is_same<coordinate<typename T::array_type>, T>;

        // free binary operators
        template<typename A>
        coordinate<A> operator+(coordinate<A> l, const coordinate<A>& r) noexcept
        {
            return std::move(l+=r);
        }
        template<typename A, typename I>
        coordinate<A> operator+(coordinate<A> l, I scalar) noexcept
        {
            return std::move(l+=scalar);
        }
        template<typename A>
        coordinate<A> operator-(coordinate<A> l, const coordinate<A>& r) noexcept
        {
            return std::move(l-=r);
        }
        template<typename A, typename I>
        coordinate<A> operator-(coordinate<A> l, I scalar) noexcept
        {
            return std::move(l-=scalar);
        }
        template<typename A>
        coordinate<A> operator%(coordinate<A> l, const coordinate<A>& r) noexcept
        {
            for (int i=0; i<coordinate<A>::size(); ++i) l[i] = l[i]%r[i];
            return l;
        }
        template<typename A, typename I>
        coordinate<A> operator%(coordinate<A> l, I scalar) noexcept
        {
            for (int i=0; i<coordinate<A>::size(); ++i) l[i] = l[i]%scalar;
            return std::move(l);
            //return std::move(l%=scalar);
        }

        template<typename A>
        coordinate<A> min(coordinate<A> l, const coordinate<A>& r) noexcept
        {
            for (int i=0; i<coordinate<A>::size(); ++i) l[i] = std::min(l[i],r[i]);
            return l;
        }
        template<typename A>
        coordinate<A> max(coordinate<A> l, const coordinate<A>& r) noexcept
        {
            for (int i=0; i<coordinate<A>::size(); ++i) l[i] = std::max(l[i],r[i]);
            return l;
        }

        template<typename A>
        auto dot(const coordinate<A>& l, const coordinate<A>& r) noexcept
        {
            auto res = l[0]*r[0];
            for (int i=1; i<A::size(); ++i) res += l[i]*r[i];
            return res;
        }

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_COMMON_COORDINATE_HPP */

