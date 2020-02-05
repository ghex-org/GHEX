#include "context_bind.hpp"
#include <array>
#include <ghex/structured/domain_descriptor.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/structured/simple_field_wrapper.hpp>
#include <ghex/communication_object_2.hpp>

struct field_descriptor {
    double *data;
    int offset[3];
    int extents[3];
    int halo[6];
    int periodic[3];
};

using field_vector_type = std::vector<field_descriptor>;
struct domain_descriptor {
    int id;
    int device_id;
    int first[3];  // indices of the first LOCAL grid point, in global index space
    int last[3];   // indices of the last LOCAL grid point, in global index space
    int gfirst[3];  // indices of the first GLOBAL grid point, (1,1,1) by default
    int glast[3];  // indices of the last GLOBAL grid point (model dimensions)
    std::unique_ptr<field_vector_type> *fields;
};

// compare two fields to establish, if the same pattern can be used for both
struct field_compare {
    bool operator()( const field_descriptor& lhs, const field_descriptor& rhs ) const
    {
        if(lhs.halo[0] < rhs.halo[0]) return true;
        if(lhs.halo[1] < rhs.halo[1]) return true;
        if(lhs.halo[2] < rhs.halo[2]) return true;
        if(lhs.halo[3] < rhs.halo[3]) return true;
        if(lhs.halo[4] < rhs.halo[4]) return true;
        if(lhs.halo[5] < rhs.halo[5]) return true;
        
        if(lhs.periodic[0] < rhs.periodic[0]) return true;
        if(lhs.periodic[1] < rhs.periodic[1]) return true;
        if(lhs.periodic[2] < rhs.periodic[2]) return true;

        return false;
    }
};

using domain_descriptor_type    = ghex::structured::domain_descriptor<int,3>;
using domain_id_type            = typename domain_descriptor_type::domain_id_type;
using grid_type                 = ghex::structured::grid;
using grid_detail_type          = ghex::structured::detail::grid<std::array<int, 3>>;
using pattern_type              = ghex::pattern_container<communicator_type, grid_detail_type, domain_id_type>;
using communication_obj_type    = ghex::communication_object<communicator_type, grid_detail_type, domain_id_type>;
using field_descriptor_type     = ghex::structured::simple_field_wrapper<double,ghex::cpu,domain_descriptor_type,2,1,0>;
using pattern_field_type        = ghex::buffer_info<pattern_type::value_type, field_descriptor_type::arch_type, field_descriptor_type>;
using pattern_field_vector_type = std::vector<pattern_field_type>;
using pattern_map_type          = std::map<field_descriptor, pattern_type, field_compare>;

extern "C"
void ghex_domain_add_field(domain_descriptor &domain_desc, field_descriptor &field_desc)
{
    if(nullptr == domain_desc.fields){
        domain_desc.fields = new std::unique_ptr<field_vector_type>( new field_vector_type() );
    }
    domain_desc.fields->get()->push_back(field_desc);
}

extern "C"
void ghex_domain_delete(domain_descriptor *domain_desc)
{
    delete domain_desc->fields;
}

extern "C"
void* ghex_exchange_new(domain_descriptor *domains_desc, int n_domains)
{

    if(0 == n_domains) return NULL;

    // For now, assume the global domains must be the same.
    // Create all necessary patterns:
    //  1. make a vector of local domain descriptors
    //  2. identify unique <halo, periodic> pairs
    //  3. make a pattern for each pair
    //  4. for each field, compute the correct pattern(wrapped_field) value

    // switch from fortran 1-based numbering to C
    std::array<int, 3> gfirst;
    gfirst[0] = domains_desc[0].gfirst[0]-1;
    gfirst[1] = domains_desc[0].gfirst[1]-1;
    gfirst[2] = domains_desc[0].gfirst[2]-1;

    std::array<int, 3> glast;
    glast[0] = domains_desc[0].glast[0]-1;
    glast[1] = domains_desc[0].glast[1]-1;
    glast[2] = domains_desc[0].glast[2]-1;

    std::vector<domain_descriptor_type> local_domains;
    for(int i=0; i<n_domains; i++){

        std::array<int, 3> first;
        first[0] = domains_desc[i].first[0]-1;
        first[1] = domains_desc[i].first[1]-1;
        first[2] = domains_desc[i].first[2]-1;

        std::array<int, 3> last;
        last[0] = domains_desc[i].last[0]-1;
        last[1] = domains_desc[i].last[1]-1;
        last[2] = domains_desc[i].last[2]-1;

        local_domains.emplace_back(domains_desc[i].id, first, last);
    }

    // a vector of `pattern(wrapped_field)` values
    pattern_field_vector_type pattern_fields;

    // a map of field descriptors to patterns
    pattern_map_type field_to_pattern;
    for(int i=0; i<n_domains; i++){
        field_vector_type &fields = *(domains_desc[i].fields->get());
        for(auto field: fields){
            auto pit = field_to_pattern.find(field);
            if (pit == field_to_pattern.end()) {
                std::array<int, 3> &periodic = *((std::array<int, 3>*)field.periodic);
                std::array<int, 6> &halo = *((std::array<int, 6>*)(field.halo));
                auto halo_generator = domain_descriptor_type::halo_generator_type(gfirst, glast, halo, periodic);
                auto pattern = ghex::make_pattern<grid_type>(*context, halo_generator, local_domains);
                field_to_pattern.emplace(std::make_pair(std::move(field), std::move(pattern)));
                pit = field_to_pattern.find(field);
            } 
            
            pattern_type &pattern = (*pit).second;
            std::array<int, 3> &offset  = *((std::array<int, 3>*)field.offset);
            std::array<int, 3> &extents = *((std::array<int, 3>*)field.extents);
            field_descriptor_type field_desc  = 
                ghex::wrap_field<ghex::cpu,2,1,0>(domains_desc[i].id, field.data, offset, extents);
            pattern_fields.push_back(pattern(field_desc));
        }
    }

    return new ghex::bindings::obj_wrapper(std::move(pattern_fields));
}

extern "C"
void ghex_exchange_delete(ghex::bindings::obj_wrapper **wrapper_ref)
{
    ghex::bindings::obj_wrapper *wrapper = *wrapper_ref;

    // clear the fortran-side variable
    *wrapper_ref = nullptr;
    delete wrapper;
}

extern "C"
void* ghex_struct_co_new()
{
    auto token = context->get_token();
    auto comm  = context->get_communicator(token);
    communication_obj_type co = ghex::make_communication_object<pattern_type>(comm);
    return new ghex::bindings::obj_wrapper(std::move(co));
}

extern "C"
void ghex_struct_co_delete(ghex::bindings::obj_wrapper **wrapper_ref)
{
    ghex::bindings::obj_wrapper *wrapper = *wrapper_ref;

    // clear the fortran-side variable
    *wrapper_ref = nullptr;
    delete wrapper;
}

extern "C"
void *ghex_struct_exchange(ghex::bindings::obj_wrapper *cowrapper, ghex::bindings::obj_wrapper *pwrapper, ghex::bindings::obj_wrapper *fwrapper)
{
    communication_obj_type &co         = *ghex::bindings::get_object_ptr_safe<communication_obj_type>(cowrapper);
    pattern_type           &pattern    = *ghex::bindings::get_object_ptr_safe<pattern_type>(pwrapper);
    field_descriptor_type  &field_desc = *ghex::bindings::get_object_ptr_safe<field_descriptor_type>(fwrapper);

    auto hex = co.exchange(pattern(field_desc));
    return new ghex::bindings::obj_wrapper(std::move(hex));
}
