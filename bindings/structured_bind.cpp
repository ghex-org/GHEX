#include "context_bind.hpp"
#include <array>
#include <ghex/structured/domain_descriptor.hpp>
#include <ghex/structured/pattern.hpp>

using domain_descriptor_type  = ghex::structured::domain_descriptor<int,3>;
using domain_id_type          = typename domain_descriptor_type::domain_id_type;
using grid_type               = ghex::structured::grid;
using pattern_type            = ghex::pattern<communicator_type, grid_type, domain_id_type>;

struct domain_descriptor {
    int id;
    int first[3];  // indices of the first LOCAL grid point, in global index space
    int last[3];   // indices of the last LOCAL grid point, in global index space
};

extern "C"
void *ghex_make_pattern(ghex::bindings::obj_wrapper *wcontext, int *_halo,
    struct domain_descriptor *_domain_desc, int ndomain_desc, int _periodic[3],
    int _g_first[3], int _g_last[3])
{
    context_type &context = wrapper2context(wcontext);

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
    auto pattern1 = ghex::make_pattern<grid_type>(context, halo1, local_domains);
    return new ghex::bindings::obj_wrapper(std::move(pattern1));
}
