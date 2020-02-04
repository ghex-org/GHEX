#include "context_bind.hpp"
#include <array>
#include <ghex/structured/domain_descriptor.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/structured/simple_field_wrapper.hpp>
#include <ghex/communication_object_2.hpp>

using domain_descriptor_type  = ghex::structured::domain_descriptor<int,3>;
using domain_id_type          = typename domain_descriptor_type::domain_id_type;
using grid_type               = ghex::structured::grid;
using grid_detail_type        = ghex::structured::detail::grid<std::array<int, 3>>;
using pattern_type            = ghex::pattern_container<communicator_type, grid_detail_type, domain_id_type>;
using communication_obj_type  = ghex::communication_object<communicator_type, grid_detail_type, domain_id_type>;
using field_descriptor_type   = ghex::structured::simple_field_wrapper<double,ghex::cpu,domain_descriptor_type,2,1,0>;

// template<typename T, typename Arch, int... Is>
// using field_descriptor_type  = ghex::structured::simple_field_wrapper<T,Arch,domain_descriptor_type, Is...>;

struct domain_descriptor {
    int id;
    int first[3];  // indices of the first LOCAL grid point, in global index space
    int last[3];   // indices of the last LOCAL grid point, in global index space
};

extern "C"
void *ghex_make_pattern(int *_halo, struct domain_descriptor *_domain_desc, int ndomain_desc, int _periodic[3],
    int _g_first[3], int _g_last[3])
{
    std::vector<domain_descriptor_type> local_domains;
    std::array<int, 3> &periodic = *((std::array<int, 3>*)_periodic);
    std::array<int, 6> &halo = *((std::array<int, 6>*)(_halo));

    // switch from fortran 1-based numbering to C
    std::array<int, 3> g_first;
    g_first[0] = _g_first[0]-1;
    g_first[1] = _g_first[1]-1;
    g_first[2] = _g_first[2]-1;

    std::array<int, 3> g_last;
    g_last[0] = _g_last[0]-1;
    g_last[1] = _g_last[1]-1;
    g_last[2] = _g_last[2]-1;

    for(int i=0; i<ndomain_desc; i++){

        std::array<int, 3> first;
        first[0] = _domain_desc[i].first[0]-1;
        first[1] = _domain_desc[i].first[1]-1;
        first[2] = _domain_desc[i].first[2]-1;

        std::array<int, 3> last;
        last[0] = _domain_desc[i].last[0]-1;
        last[1] = _domain_desc[i].last[1]-1;
        last[2] = _domain_desc[i].last[2]-1;

        local_domains.emplace_back(_domain_desc[i].id, first, last);
    }

    auto halo1    = domain_descriptor_type::halo_generator_type(g_first, g_last, halo, periodic);

    pattern_type pattern = ghex::make_pattern<grid_type>(*context, halo1, local_domains);
    return new ghex::bindings::obj_wrapper(std::move(pattern));
}

extern "C"
void *ghex_wrap_field(int domain_id, double *field, int _local_offset[3], int _field_extents[3])
{
    std::array<int, 3> &local_offset  = *((std::array<int, 3>*)_local_offset);
    std::array<int, 3> &field_extents = *((std::array<int, 3>*)_field_extents);
    field_descriptor_type field_desc  = ghex::wrap_field<ghex::cpu,2,1,0>(domain_id, field, local_offset, field_extents);
    return new ghex::bindings::obj_wrapper(std::move(field_desc));
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
