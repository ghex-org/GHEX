#include "context_bind.hpp"
#include <array>
#include <ghex/bulk_communication_object.hpp>
#include <ghex/structured/rma_range.hpp>
#include <ghex/structured/rma_range_generator.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/structured/pattern.hpp>

// those are configurable at compile time
#include "ghex_defs.hpp"
using arch_type                 = ghex::cpu;
using domain_id_type            = int;

struct struct_field_descriptor {
    fp_type *data;
    int   offset[3];
    int  extents[3];
    int     halo[6];
    int periodic[3];
    int n_components;
    int layout;
};

using field_vector_type = std::vector<struct_field_descriptor>;
struct struct_domain_descriptor {
    field_vector_type *fields;
    int id;
    int device_id;
    int  first[3];  // indices of the first LOCAL grid point, in global index space
    int   last[3];  // indices of the last LOCAL grid point, in global index space
    int gfirst[3];  // indices of the first GLOBAL grid point, (1,1,1) by default
    int  glast[3];  // indices of the last GLOBAL grid point (model dimensions)
};


using grid_type                 = ghex::structured::grid;
using grid_detail_type          = ghex::structured::detail::grid<std::array<int, GHEX_DIMS>>;
using domain_descriptor_type    = ghex::structured::regular::domain_descriptor<domain_id_type, GHEX_DIMS>;
using pattern_type              = ghex::pattern_container<communicator_type, grid_detail_type, domain_id_type>;
using communication_obj_type    = ghex::communication_object<communicator_type, grid_detail_type, domain_id_type>;
using field_descriptor_type     = ghex::structured::regular::field_descriptor<fp_type, arch_type, domain_descriptor_type,2,1,0>;
using exchange_handle_type      = communication_obj_type::handle_type;
using bco_type                  = ghex::bulk_communication_object<ghex::structured::rma_range_generator, pattern_type, field_descriptor_type>;
using buffer_info_type          = bco_type::buffer_info_type<field_descriptor_type>;

// ASYMETRY
using halo_generator_type       = ghex::structured::regular::halo_generator<domain_id_type, GHEX_DIMS>;

extern "C"
void ghex_struct_co_init(ghex::bindings::obj_wrapper **wrapper_ref)
{
    // TODO: is this still relevant? tokens..
    // auto token = context->get_token();
    // auto comm  = context->get_communicator(token);
    // *wrapper_ref = new ghex::bindings::obj_wrapper(ghex::make_communication_object<pattern_type>(comm));
    *wrapper_ref = NULL;
}

extern "C"
void ghex_struct_domain_add_field(struct_domain_descriptor *domain_desc, struct_field_descriptor *field_desc)
{
    if(nullptr == domain_desc->fields){
        domain_desc->fields = new field_vector_type();
    } else {
	// TODO: verify that periodicity and halos are the same for the added field
	const struct_field_descriptor &fd = domain_desc->fields->front();
	for(int i=0; i<6; i++)
	    if(fd.halo[i] != field_desc->halo[i]){
		std::cerr << "Bad halo definition: when constructing a bulk exchange domain, all fields must have the same halo and periodicity definition.";
		std::terminate();
	    }
	for(int i=0; i<3; i++)
	    if(fd.periodic[i] != field_desc->periodic[i]){
		std::cerr << "Bad periodicity definition: when constructing a bulk exchange domain, all fields must have the same halo and periodicity definition.";
		std::terminate();
	    }
    }
    domain_desc->fields->push_back(*field_desc);
}

extern "C"
void ghex_struct_domain_free(struct_domain_descriptor *domain_desc)
{
    delete domain_desc->fields;
    domain_desc->fields = nullptr;
    domain_desc->id = -1;
    domain_desc->device_id = -1;
}

extern "C"
void* ghex_struct_exchange_desc_new(struct_domain_descriptor *domains_desc, int n_domains)
{

    if(0 == n_domains) return NULL;

    // Create all necessary patterns:
    //  1. make a vector of local domain descriptors
    //  2. identify unique <halo, periodic> pairs
    //  3. make a pattern for each pair
    //  4. for each field, compute the correct pattern(wrapped_field) objects

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

    // halo and periodicity must be the same
    std::array<int, 3> &periodic = *((std::array<int, 3>*)domains_desc[0].fields->front().periodic);
    std::array<int, 6> &halo = *((std::array<int, 6>*)domains_desc[0].fields->front().halo);
    auto halo_generator = halo_generator_type(gfirst, glast, halo, periodic);
    auto pattern = ghex::make_pattern<grid_type>(*context, halo_generator, local_domains);

    // TODO: is this still relevant? or do I need comm as argument?
    auto token = context->get_token();
    auto comm  = context->get_communicator(token);
    
    auto bco = gridtools::ghex::bulk_communication_object<
	gridtools::ghex::structured::rma_range_generator,
	pattern_type,
	field_descriptor_type
	> (comm);

    for(int i=0; i<n_domains; i++){
        field_vector_type &fields = *(domains_desc[i].fields);
        for(auto field: fields){
            std::array<int, 3> &offset  = *((std::array<int, 3>*)field.offset);
            std::array<int, 3> &extents = *((std::array<int, 3>*)field.extents);
	    auto f = field_descriptor_type(local_domains[i], field.data, offset, extents, field.n_components, false, 
					   domains_desc[i].device_id);
	    bco.add_field(pattern(f));
        }
    }

    // exchange the RMA handles before any other BCO can be created
    // bco.init();
    return new ghex::bindings::obj_wrapper(std::move(bco));
}

extern "C"
void *ghex_struct_exchange(ghex::bindings::obj_wrapper *cowrapper, ghex::bindings::obj_wrapper *ewrapper)
{
    if(nullptr == ewrapper) return nullptr;
    return new ghex::bindings::obj_wrapper(ghex::bindings::get_object_ptr_safe<bco_type>(ewrapper)->exchange());
}

extern "C"
void ghex_struct_exchange_handle_wait(ghex::bindings::obj_wrapper **ehwrapper)
{
    if(nullptr == *ehwrapper) return;
    exchange_handle_type &hex = *ghex::bindings::get_object_ptr_safe<exchange_handle_type>(*ehwrapper);
    hex.wait();
    *ehwrapper = nullptr;
}
