/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "context_bind.hpp"
#include <array>
#include <ghex/structured/cubed_sphere/domain_descriptor.hpp>
#include <ghex/structured/cubed_sphere/halo_generator.hpp>
#include <ghex/structured/cubed_sphere/field_descriptor.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/communication_object_2.hpp>

// those are configurable at compile time
#include "ghex_defs.hpp"

using namespace gridtools::ghex::fhex;
using arch_type                 = ghex::cpu;
using domain_id_type            = ghex::structured::cubed_sphere::domain_id_type;

namespace gridtools {
    namespace ghex {
        namespace fhex {

            struct cubed_sphere_field_descriptor {
                fp_type *data;
                int   offset[3];
                int  extents[3];
                int     halo[4];
                int n_components;
                int layout;
                bool is_vector;
            };

            using field_vector_type = std::vector<cubed_sphere_field_descriptor>;
            struct cubed_sphere_domain_descriptor {
                field_vector_type *fields = nullptr;
                int tile;
                int device_id;
                int   cube[2];  // local grid dimensions
                int  first[2];  // indices of the first LOCAL grid point, in global index space
                int   last[2];  // indices of the last LOCAL grid point, in global index space
            };

            // compare two fields to establish, if the same pattern can be used for both
            struct field_compare {
                bool operator()( const cubed_sphere_field_descriptor& lhs, const cubed_sphere_field_descriptor& rhs ) const
                {
                    if(lhs.halo[0] < rhs.halo[0]) return true;
                    if(lhs.halo[0] > rhs.halo[0]) return false;
                    if(lhs.halo[1] < rhs.halo[1]) return true;
                    if(lhs.halo[1] > rhs.halo[1]) return false;
                    if(lhs.halo[2] < rhs.halo[2]) return true;
                    if(lhs.halo[2] > rhs.halo[2]) return false;
                    if(lhs.halo[3] < rhs.halo[3]) return true;
                    if(lhs.halo[3] > rhs.halo[3]) return false;

                    return false;
                }
            };

            using grid_type                 = ghex::structured::grid;
            using grid_detail_type          = ghex::structured::detail::grid<ghex::coordinate<std::array<int, 4>>>; // only 3D grids
            using domain_descriptor_type    = ghex::structured::cubed_sphere::domain_descriptor;
            using pattern_type              = ghex::pattern_container<communicator_type, grid_detail_type, domain_id_type>;
            using communication_obj_type    = ghex::communication_object<communicator_type, grid_detail_type, domain_id_type>;
            using pattern_map_type          = std::map<cubed_sphere_field_descriptor, pattern_type, field_compare>;
            using exchange_handle_type      = communication_obj_type::handle_type;
            using halo_generator_type       = ghex::structured::cubed_sphere::halo_generator;

            // row-major storage
            using field_descriptor_type_1     = ghex::structured::cubed_sphere::field_descriptor<fp_type, arch_type, ::gridtools::layout_map<3,2,1,0>>;
            using pattern_field_type_1        = ghex::buffer_info<pattern_type::value_type, arch_type, field_descriptor_type_1>;
            using pattern_field_vector_type_1 = std::pair<std::vector<std::unique_ptr<field_descriptor_type_1>>, std::vector<pattern_field_type_1>>;

            // field-major storage
            using field_descriptor_type_2     = ghex::structured::cubed_sphere::field_descriptor<fp_type, arch_type, ::gridtools::layout_map<2,1,0,3>>;
            using pattern_field_type_2        = ghex::buffer_info<pattern_type::value_type, arch_type, field_descriptor_type_2>;
            using pattern_field_vector_type_2 = std::pair<std::vector<std::unique_ptr<field_descriptor_type_2>>, std::vector<pattern_field_type_2>>;

            struct pattern_field_data {
                pattern_field_vector_type_1 row_major;
                pattern_field_vector_type_2 field_major;
            };

            // a map of field descriptors to patterns
            static pattern_map_type field_to_pattern;
        }
    }
}

extern "C"
void ghex_cubed_sphere_co_init(obj_wrapper **wco_ref, obj_wrapper *wcomm)
{
    if(nullptr == wcomm) return;   
    auto &comm = *get_object_ptr_unsafe<communicator_type>(wcomm);
    *wco_ref = new obj_wrapper(ghex::make_communication_object<pattern_type>(comm));
}

extern "C"
void ghex_cubed_sphere_domain_add_field(cubed_sphere_domain_descriptor *domain_desc, cubed_sphere_field_descriptor *field_desc)
{
    if(nullptr == domain_desc || nullptr == field_desc) return;
    if(nullptr == domain_desc->fields){
        domain_desc->fields = new field_vector_type();
    }
    domain_desc->fields->push_back(*field_desc);
}

extern "C"
void ghex_cubed_sphere_domain_free(cubed_sphere_domain_descriptor *domain_desc)
{
    if(nullptr == domain_desc) return;
    delete domain_desc->fields;
    domain_desc->fields = nullptr;
    domain_desc->tile = -1;
    domain_desc->device_id = -1;
}

extern "C"
void* ghex_cubed_sphere_exchange_desc_new(cubed_sphere_domain_descriptor *domains_desc, int n_domains)
{

    if(0 == n_domains || nullptr == domains_desc) return nullptr;

    // Create all necessary patterns:
    //  1. make a vector of local domain descriptors
    //  2. identify fields with unique halos
    //  3. make a pattern for each type of field
    //  4. for each field, compute the correct pattern(field) object

    // switch from fortran 1-based numbering to C
    std::vector<domain_descriptor_type> local_domains;
    for(int i=0; i<n_domains; i++){
        ghex::structured::cubed_sphere::cube c = {domains_desc[i].cube[0], domains_desc[i].cube[1]};
        local_domains.emplace_back(c, domains_desc[i].tile, 
            domains_desc[i].first[0]-1, domains_desc[i].last[0]-1,
            domains_desc[i].first[1]-1, domains_desc[i].last[1]-1);
    }

    // a vector of `pattern(field)` objects
    pattern_field_data pattern_fields;

    for(int i=0; i<n_domains; i++){

        if(nullptr == domains_desc[i].fields) continue;
        
        field_vector_type &fields = *(domains_desc[i].fields);
        for(auto field: fields){
            auto pit = field_to_pattern.find(field);
            if (pit == field_to_pattern.end()) {
                std::array<int, 4> &halo = *((std::array<int, 4>*)(field.halo));
                auto halo_generator = halo_generator_type(halo);
                pit = field_to_pattern.emplace(std::make_pair(std::move(field),
                        ghex::make_pattern<grid_type>(*ghex_context, halo_generator, local_domains))).first;
            }

            pattern_type &pattern = (*pit).second;
            std::array<int, 3> &offset  = *((std::array<int, 3>*)field.offset);
            std::array<int, 3> &extents = *((std::array<int, 3>*)field.extents);
            // ASYMETRY

	    if(GhexLayoutFieldLast == field.layout){
		std::unique_ptr<field_descriptor_type_1> field_desc_uptr(new field_descriptor_type_1(local_domains[i], field.data, offset, extents, field.n_components, field.is_vector));
		auto ptr = field_desc_uptr.get();
		pattern_fields.row_major.first.push_back(std::move(field_desc_uptr));
		pattern_fields.row_major.second.push_back(pattern(*ptr));
	    } else {
		std::unique_ptr<field_descriptor_type_2> field_desc_uptr(new field_descriptor_type_2(local_domains[i], field.data, offset, extents, field.n_components, field.is_vector));
		auto ptr = field_desc_uptr.get();
		pattern_fields.field_major.first.push_back(std::move(field_desc_uptr));
		pattern_fields.field_major.second.push_back(pattern(*ptr));
	    }
        }
    }

    return new obj_wrapper(std::move(pattern_fields));
}

extern "C"
void *ghex_cubed_sphere_exchange(obj_wrapper *cowrapper, obj_wrapper *ewrapper)
{
    if(nullptr == cowrapper || nullptr == ewrapper) return nullptr;
    communication_obj_type    &co      = *get_object_ptr_unsafe<communication_obj_type>(cowrapper);
    pattern_field_data &pattern_fields = *get_object_ptr_unsafe<pattern_field_data>(ewrapper);
    return new obj_wrapper(co.exchange(pattern_fields.row_major.second.begin(),
						       pattern_fields.row_major.second.end(),
						       pattern_fields.field_major.second.begin(),
						       pattern_fields.field_major.second.end()));
}

extern "C"
void ghex_cubed_sphere_exchange_handle_wait(obj_wrapper **ehwrapper)
{
    if(nullptr == *ehwrapper) return;
    exchange_handle_type &hex = *get_object_ptr_unsafe<exchange_handle_type>(*ehwrapper);
    hex.wait();
    *ehwrapper = nullptr;
}
