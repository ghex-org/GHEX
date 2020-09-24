#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <future>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
#define TRANSPORT tl::mpi_tag
#else
#include <ghex/transport_layer/ucx/context.hpp>
#define TRANSPORT tl::ucx_tag
#endif
#include <ghex/threads/std_thread/primitives.hpp>
#include <ghex/bulk_communication_object.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/structured/rma_range_generator.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>

using namespace gridtools::ghex;
using arr       = std::array<int,2>;
using transport = TRANSPORT;
using threading = threads::std_thread::primitives;
using factory   = tl::context_factory<transport,threading>;
using domain    = structured::regular::domain_descriptor<int,2>;
using halo_gen  = structured::regular::halo_generator<int,2>;

#define DIM 8
#define HALO 3
#define PX true
#define PY true

constexpr std::array<int,4>  halos{HALO,HALO,HALO,HALO};
constexpr std::array<bool,2> periodic{PX,PY};

std::vector<arr> allocate_field()
{ 
    return {(HALO*2+DIM) * (HALO*2+DIM/2), std::array<int,2>{-1,-1}};
}

template<typename RawField>
auto wrap_field(RawField& raw_field, const domain& d)
{
    return wrap_field<cpu,1,0>(d, raw_field.data(), arr{HALO, HALO}, arr{HALO*2+DIM, HALO*2+DIM/2});
}

template<typename Field>
Field&& fill(Field&& field)
{
    for (int j=0; j<DIM/2; ++j)
        for (int i=0; i<DIM; ++i)
        {
            auto& v = field({i,j});
            v[0] = field.domain().first()[0]+i;
            v[1] = field.domain().first()[1]+j;
        }
    return std::forward<Field>(field);
}

template<typename Field>
Field&& reset(Field&& field)
{
    for (int j=-HALO; j<DIM/2+HALO; ++j)
        for (int i=-HALO; i<DIM+HALO; ++i)
        {
            auto& v = field({i,j});
            v[0] = -1;
            v[1] = -1;
        }
    return fill(std::forward<Field>(field));
}

int expected(int coord, int dim, int first, int last, bool periodic_)
{
    const auto ext = (last+1) - first;
    int res = first+coord;
    if (first == 0 && coord < 0) res = periodic_ ? dim*DIM + coord : -1;
    if (last == dim*DIM-1 && coord >= ext) res = periodic_ ? coord - ext : -1;
    return res;
}

bool compare(const arr& v, int x, int y)
{
    if (x == -1 || y == -1) { x = -1; y = -1; }
    return (x == v[0]) && (y == v[1]);
} 

template<typename Field>
bool check(const Field& field, const arr& dims)
{
    bool res = true;
    for (int j=-HALO; j<DIM/2+HALO; ++j)
    {
        const auto y = expected(j, dims[1], field.domain().first()[1],
            field.domain().last()[1], periodic[1]);
        for (int i=-HALO; i<DIM+HALO; ++i)
        {
            const auto x = expected(i, dims[0], field.domain().first()[0],
                field.domain().last()[0], periodic[0]);
            res = res && compare(field({i,j}),x,y);
        }
    }
    return res;
}

auto make_domain(int rank, int id, std::array<int,2> coord)
{
    const auto x = coord[0]*DIM;
    const auto y = (coord[1]*2+id)*(DIM/2);
    return domain{rank*2+id, arr{x, y}, arr{x+DIM-1, y+DIM/2-1}};
}

template<typename Context, typename Pattern, typename Domains>
bool run(Context& context, const Pattern& pattern, const Domains& domains, const arr& dims, int thread_id)
{
    bool res = true;
    // field
    auto raw_field = allocate_field();
    auto field     = fill(wrap_field(raw_field, domains[thread_id]));
    // get a communcator
    auto comm = context.get_communicator(context.get_token());
    
    // general exchange
    // ================
    auto co = make_communication_object<Pattern>(comm);
    auto h0 = co.exchange(pattern(field));
    // ...
    h0.wait();

    // check field
    res = res && check(field, dims);

    // reset field
    reset(field);

    // bulk exchange (rma)
    // ===================
    auto bco = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field)>(comm);
    bco.add_field(pattern(field));
    auto h1 = bco.exchange();
    // ...
    h1.wait();

    // check field
    res = res && check(field, dims);
    return res;
}

template<typename Context, typename Pattern, typename Domains>
bool run(Context& context, const Pattern& pattern, const Domains& domains, const arr& dims)
{
    bool res = true;
    // fields
    auto raw_field_a = allocate_field();
    auto raw_field_b = allocate_field();
    auto field_a     = fill(wrap_field(raw_field_a, domains[0]));
    auto field_b     = fill(wrap_field(raw_field_b, domains[1]));
    // get a communcator
    auto comm = context.get_communicator(context.get_token());
    
    // general exchange
    // ================
    auto co = make_communication_object<Pattern>(comm);
    auto h0 = co.exchange(pattern(field_a), pattern(field_b));
    // ...
    h0.wait();

    // check fields
    res = res && check(field_a, dims);
    res = res && check(field_b, dims);

    // reset fields
    reset(field_a);
    reset(field_b);

    // bulk exchange (rma)
    // ===================
    auto bco = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field_a)>(comm);
    bco.add_field(pattern(field_a));
    bco.add_field(pattern(field_b));
    auto h1 = bco.exchange();
    // ...
    h1.wait();

    // check fields
    res = res && check(field_a, dims);
    res = res && check(field_b, dims);
    return res;
}

void sim(bool multi_threaded)
{
    // make a context from mpi world and number of threads
    auto context_ptr = factory::create(multi_threaded? 2 : 1, MPI_COMM_WORLD);
    auto& context    = *context_ptr;
    // 2D domain decomposition
    arr dims{0,0}, coords{0,0};
    MPI_Dims_create(context.size(), 2, dims.data());
    coords[1] = context.rank()/dims[0];
    coords[0] = context.rank() - coords[1]*dims[0];
    // make 2 domains per rank
    std::vector<domain> domains{
        make_domain(context.rank(), 0, coords),
        make_domain(context.rank(), 1, coords)};
    // make halo generator
    halo_gen gen{arr{0,0}, arr{dims[0]*DIM-1,dims[1]*DIM-1}, halos, periodic};
    // create a pattern for communication
    auto pattern = make_pattern<structured::grid>(context, gen, domains);
    // run
    bool res = true;
    if (multi_threaded)
    {
        auto run_fct = [&context,&pattern,&domains,&dims](int id)
            { return run(context, pattern, domains, dims, id); };
        auto f1 = std::async(std::launch::async, run_fct, 0);
        auto f2 = std::async(std::launch::async, run_fct, 1);
        res = res && f1.get();
        res = res && f2.get();
    }
    else
    {
        res = res && run(context, pattern, domains, dims);
    }
    // reduce res
    bool all_res = false;
    MPI_Reduce(&res, &all_res, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
    if (context.rank() == 0)
    {
        EXPECT_TRUE(all_res);
    }
}

TEST(simple_regular_exchange, single)
{
    sim(false);
}

TEST(simple_regular_exchange, multi)
{
    sim(false);
}

