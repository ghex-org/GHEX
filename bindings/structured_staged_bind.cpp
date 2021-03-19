#include "context_bind.hpp"
#include <array>
#include <ghex/bulk_communication_object.hpp>
#include <ghex/structured/rma_range.hpp>
#include <ghex/structured/rma_range_generator.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/structured/regular/make_pattern.hpp>
#include <ghex/structured/pattern.hpp>

extern "C" {
#include <hwcart/hwcart.h>
}

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
    int  first[3];   // indices of the first LOCAL grid point, in global index space
    int   last[3];   // indices of the last LOCAL grid point, in global index space
    int gfirst[3];   // indices of the first GLOBAL grid point, (1,1,1) by default
    int  glast[3];   // indices of the last GLOBAL grid point (model dimensions)

    // cartesian communicator info
    int cart_comm;   // Fortran-side cartesian communicator (either hwcart, or mpi_cart)
    hwcart_order_t cart_order;  // dimension order for rank2coord and coord2rank calculations (for hwcart only)
    int cart_dim[3]; // global rank space dimensions
};

// compare two fields to establish, if the same pattern can be used for both
struct field_compare {
    bool operator()( const struct_field_descriptor& lhs, const struct_field_descriptor& rhs ) const
    {
        if(lhs.halo[0] < rhs.halo[0]) return true; 
        if(lhs.halo[0] > rhs.halo[0]) return false;
        if(lhs.halo[1] < rhs.halo[1]) return true; 
        if(lhs.halo[1] > rhs.halo[1]) return false;
        if(lhs.halo[2] < rhs.halo[2]) return true; 
        if(lhs.halo[2] > rhs.halo[2]) return false;
        if(lhs.halo[3] < rhs.halo[3]) return true; 
        if(lhs.halo[3] > rhs.halo[3]) return false;
        if(lhs.halo[4] < rhs.halo[4]) return true; 
        if(lhs.halo[4] > rhs.halo[4]) return false;
        if(lhs.halo[5] < rhs.halo[5]) return true; 
        if(lhs.halo[5] > rhs.halo[5]) return false;
        
        if(lhs.periodic[0] < rhs.periodic[0]) return true; 
        if(lhs.periodic[0] > rhs.periodic[0]) return false;
        if(lhs.periodic[1] < rhs.periodic[1]) return true; 
        if(lhs.periodic[1] > rhs.periodic[1]) return false;
        if(lhs.periodic[2] < rhs.periodic[2]) return true; 
        if(lhs.periodic[2] > rhs.periodic[2]) return false;

        return false;
    }
};


using grid_type                 = ghex::structured::grid;
using grid_detail_type          = ghex::structured::detail::grid<std::array<int, GHEX_DIMS>>;
using domain_descriptor_type    = ghex::structured::regular::domain_descriptor<domain_id_type, GHEX_DIMS>;
using pattern_type              = ghex::pattern_container<communicator_type, grid_detail_type, domain_id_type>;
using communication_obj_type    = ghex::communication_object<communicator_type, grid_detail_type, domain_id_type>;
using field_descriptor_type     = ghex::structured::regular::field_descriptor<fp_type, arch_type, domain_descriptor_type,2,1,0>;
using pattern_field_type        = ghex::buffer_info<pattern_type::value_type, arch_type, field_descriptor_type>;
using pattern_field_vector_type = std::pair<std::vector<std::unique_ptr<field_descriptor_type>>, std::vector<pattern_field_type>>;
using stage_patterns_type       = std::array<std::unique_ptr<pattern_type>, 3>;
using pattern_map_type          = std::map<struct_field_descriptor, stage_patterns_type, field_compare>;
using bco_type                  = ghex::bulk_communication_object<ghex::structured::rma_range_generator, pattern_type, field_descriptor_type>;
using exchange_handle_type      = bco_type::handle;
using buffer_info_type          = bco_type::buffer_info_type<field_descriptor_type>;
using halo_generator_type       = ghex::structured::regular::halo_generator<domain_id_type, GHEX_DIMS>;

// a map of field descriptors to patterns
static pattern_map_type field_to_pattern;

// each bulk communication object can only be used with
// one set of domains / fields. Since the fortran API
// allows passing different exchange handles to the ghex_exchange
// function, we have to check if the BCO and EH match.
struct bco_wrapper {
  bco_type bco_x;
  bco_type bco_y;
  bco_type bco_z;
  void *eh;
};

extern "C"
void ghex_struct_co_init(ghex::bindings::obj_wrapper **wco_ref, ghex::bindings::obj_wrapper *wcomm)
{
    if(nullptr == wcomm) return;   
    auto &comm = *ghex::bindings::get_object_ptr_unsafe<communicator_type>(wcomm);

    auto bco_x = gridtools::ghex::bulk_communication_object<
        gridtools::ghex::structured::rma_range_generator,
        pattern_type,
        field_descriptor_type
        > (comm);
    auto bco_y = gridtools::ghex::bulk_communication_object<
        gridtools::ghex::structured::rma_range_generator,
        pattern_type,
        field_descriptor_type
        > (comm);
    auto bco_z = gridtools::ghex::bulk_communication_object<
        gridtools::ghex::structured::rma_range_generator,
        pattern_type,
        field_descriptor_type
        > (comm);
    
    *wco_ref = new ghex::bindings::obj_wrapper(bco_wrapper{std::move(bco_x),std::move(bco_y),std::move(bco_z),NULL});
}

extern "C"
void ghex_struct_domain_add_field(struct_domain_descriptor *domain_desc, struct_field_descriptor *field_desc)
{
    if(nullptr == domain_desc || nullptr == field_desc) return;
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
    if(nullptr == domain_desc) return;
    delete domain_desc->fields;
    domain_desc->fields = nullptr;
    domain_desc->id = -1;
    domain_desc->device_id = -1;
}

extern "C"
void* ghex_struct_exchange_desc_new(struct_domain_descriptor *domains_desc, int n_domains)
{
    if(0 == n_domains || nullptr == domains_desc) return NULL;

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

        if(0 == domains_desc[i].cart_comm){
            std::cerr << "The staged communicator requires cart_comm, cart_order, and cart_dim info in the domain descriptor." << std::endl;
            std::terminate();
        }

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
   
    // a vector of `pattern(field)` objects
    std::array<pattern_field_vector_type,3> pattern_fields_array;

    for(int i=0; i<n_domains; i++){
        field_vector_type &fields = *(domains_desc[i].fields);
        auto &domain_desc = domains_desc[i];
        std::array<int, 3> &cart_dim = *((std::array<int, 3>*)domain_desc.cart_dim);
        MPI_Comm cart_comm = MPI_Comm_f2c(domain_desc.cart_comm);
        if (MPI_COMM_NULL == cart_comm){
            std::cerr << "Illegal cartesian communicator " << domain_desc.cart_comm << std::endl;
            std::terminate();
        }
        for(auto field: fields){
            auto pit = field_to_pattern.find(field);
            if (pit == field_to_pattern.end()) {
                std::array<bool, 3> periodic;
                periodic[0] = field.periodic[0]!=0;
                periodic[1] = field.periodic[1]!=0;
                periodic[2] = field.periodic[2]!=0;
                
                std::array<int, 6> &halo = *((std::array<int, 6>*)field.halo);
                auto halo_generator = halo_generator_type(gfirst, glast, halo, periodic);

                auto pattern = ghex::structured::regular::make_staged_pattern(
                    *context, local_domains,                    
                    [&domain_desc,&cart_comm,&field](auto id, auto const& offset) {
                        int coord[3], nbrank;
                        
                        // NOTE: we assume domain id is the same as rank
                        hwcart_rank2coord(cart_comm, domain_desc.cart_dim, id, domain_desc.cart_order, coord);
                        coord[0] += offset[0];
                        coord[1] += offset[1];
                        coord[2] += offset[2];
                        hwcart_coord2rank(cart_comm, domain_desc.cart_dim, field.periodic, coord, domain_desc.cart_order, &nbrank);
                        struct _neighbor
                        {
                            int m_id;
                            int m_rank;
                            int id() const noexcept { return m_id; }
                            int rank() const noexcept { return m_rank; }
                        };
                        return _neighbor{nbrank, nbrank};
                    },
                    gfirst,
                    glast,
                    halo,
                    periodic);

                pit = field_to_pattern.emplace(std::make_pair(std::move(field), std::move(pattern))).first;
            } 
            
            stage_patterns_type &pattern = (*pit).second;
            std::array<int, 3> &offset  = *((std::array<int, 3>*)field.offset);
            std::array<int, 3> &extents = *((std::array<int, 3>*)field.extents);
            std::unique_ptr<field_descriptor_type>
                field_desc_uptr(new field_descriptor_type(ghex::wrap_field<arch_type,2,1,0>(local_domains[i], field.data, offset, extents)));
            auto ptr = field_desc_uptr.get();

            // keep pointer around
            pattern_fields_array[0].first.push_back(std::move(field_desc_uptr));

            // apply stage patterns to the field
            pattern_fields_array[0].second.push_back(pattern[0]->operator()(*ptr));
            pattern_fields_array[1].second.push_back(pattern[1]->operator()(*ptr));
            pattern_fields_array[2].second.push_back(pattern[2]->operator()(*ptr));            
        }
    }

    return new ghex::bindings::obj_wrapper(std::move(pattern_fields_array));
}

extern "C"
void *ghex_struct_exchange(ghex::bindings::obj_wrapper *cowrapper, ghex::bindings::obj_wrapper *ewrapper)
{
    if(nullptr == cowrapper || nullptr == ewrapper) return nullptr;
    
    bco_wrapper &bcowr = *ghex::bindings::get_object_ptr_unsafe<bco_wrapper>(cowrapper);
    std::array<pattern_field_vector_type,3> &pattern_fields_array =
        *ghex::bindings::get_object_ptr_unsafe<std::array<pattern_field_vector_type,3>>(ewrapper);

    // first time call: build the bco
    if(!bcowr.eh) {
        for (auto it=pattern_fields_array[0].second.begin(); it!=pattern_fields_array[0].second.end(); ++it) {
            bcowr.bco_x.add_field(*it);
        }
        for (auto it=pattern_fields_array[1].second.begin(); it!=pattern_fields_array[1].second.end(); ++it) {
            bcowr.bco_y.add_field(*it);
        }
        for (auto it=pattern_fields_array[2].second.begin(); it!=pattern_fields_array[2].second.end(); ++it) {
            bcowr.bco_z.add_field(*it);
        }

        // exchange the RMA handles before any other BCO can be created
        bcowr.bco_x.init();
        bcowr.bco_y.init();
        bcowr.bco_z.init();
	bcowr.eh = ewrapper;
    } else {
        if(bcowr.eh != ewrapper){
	  std::cerr << "This RMA communication object was previously initialized and used with different fields. You have to create a new communication object for this new set of fields." << std::endl;
	  std::terminate();
        }
    }

    bcowr.bco_x.exchange().wait();
    bcowr.bco_y.exchange().wait();
    return new ghex::bindings::obj_wrapper(bcowr.bco_z.exchange());
}

extern "C"
void ghex_struct_exchange_handle_wait(ghex::bindings::obj_wrapper **ehwrapper)
{
    if(nullptr == *ehwrapper) return;
    exchange_handle_type &hex = *ghex::bindings::get_object_ptr_unsafe<exchange_handle_type>(*ehwrapper);
    hex.wait();
}
