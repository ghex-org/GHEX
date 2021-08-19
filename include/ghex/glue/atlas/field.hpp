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
#ifndef INCLUDED_GHEX_GLUE_ATLAS_USER_CONCEPTS_HPP
#define INCLUDED_GHEX_GLUE_ATLAS_USER_CONCEPTS_HPP

#include <utility>

#ifdef GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_KFIRST
#include <gridtools/storage/cpu_kfirst.hpp>
#endif
#ifdef GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_IFIRST
#include <gridtools/storage/cpu_ifirst.hpp>
#endif
#ifdef GHEX_ATLAS_GT_STORAGE_GPU_BACKEND
#include <gridtools/storage/gpu.hpp>
#endif

#include <atlas/functionspace/NodeColumns.h>


namespace gridtools {

    namespace ghex {

        namespace atlas {

            using idx_t = atlas::idx_t;

            template <std::size_t N>
            struct dims;

            template <>
            struct dims<2> {
                idx_t x, y;
            }

            template <>
            struct dims<3> {
                idx_t x, y, z;
            }

            // TO DO: might be used in different implementations
            template <typename T, typename StorageTraits>
            inline auto 2d_storage_builder(const dims<2>& d) {
                using value_type = T;
                using storage_traits = StorageTraits;
                return gridtools::storage::builder<storage_traits>
                    .type<value_type>()
                    .dimensions(d.x, d.y);
            }

            // TO DO: dimension should be generalized
            template <typename T, typename StorageTraits>
            inline auto 3d_storage_builder(const dims<3>& d) {
                using value_type = T;
                using storage_traits = StorageTraits;
                return gridtools::storage::builder<storage_traits>
                    .type<value_type>()
                    .dimensions(d.x, d.y, d.z);
            }

            template <typename T, typename StorageTraits, typename FunctionSpace>
            class field;

            template <typename T, typename StorageTraits>
            class field<T, StorageTraits, atlas::functionspace::NodeColumns> {

                public:

                    using value_type = T;
                    using storage_traits = StorageTraits;
                    using function_space_type = atlas::functionspace::NodeColumns;

                private:
                    
                    // TO DO: 3d storage is hard-coded. That might not be optimal i.e. for scalar fields, or 2d fields (levels = 1)
                    using storage_type = decltype(3d_storage_builder<value_type, storage_traits>(std::declval<dims<3>>())());

                    storage_type m_st;
                    const function_space_type& m_fs;
                    idx_t m_components;

                public:

                    field(const function_space_type& fs, idx_t components = 1) : m_fs{fs}, m_components{components} {
                        idx_t x{fs.nb_nodes()};
                        idx_t y{fs.levels()};
                        idx_t z{components};
                        dims<3> d{x, y, z};
                        m_st = 3d_storage_builder<value_type, storage_traits>(d)();
                    }

                    idx_t components() const noexcept { return m_components; }
                    
                    auto host_view() { return m_st->host_view(); }
                    auto const_host_view() { return m_st->const_host_view(); }
                    auto target_view() { return m_st->target_view(); }
                    auto const_target_view() { return m_st->const_target_view(); }

            };

        } // namespace atlas

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_GLUE_ATLAS_USER_CONCEPTS_HPP */
