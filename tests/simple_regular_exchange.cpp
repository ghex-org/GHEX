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
#include <ghex/transport_layer/util/barrier.hpp>
#include <ghex/bulk_communication_object.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/structured/rma_range_generator.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <ghex/structured/regular/make_pattern.hpp>
#include <ghex/cuda_utils/error.hpp>
#include <gridtools/common/array.hpp>

#include <ghex/common/defs.hpp>
#ifdef GHEX_CUDACC
#include <ghex/common/cuda_runtime.hpp>
#endif

using namespace gridtools::ghex;
using arr       = std::array<int,2>;
using transport = TRANSPORT;
using factory   = tl::context_factory<transport>;
using domain    = structured::regular::domain_descriptor<int,std::integral_constant<int, 2>>;
using halo_gen  = structured::regular::halo_generator<int,std::integral_constant<int, 2>>;

#define DIM 8
#define HALO 3
#define PX true
#define PY true

constexpr std::array<int,4>  halos{HALO,HALO,HALO,HALO};
constexpr std::array<bool,2> periodic{PX,PY};

template<typename T>
struct memory
{
    unsigned int m_size;
    std::unique_ptr<T[]> m_host_memory;
#ifdef GHEX_CUDACC
    struct cuda_deleter
    {
        // no delete since this messes up the rma stuff
        // when doing 2 tests in a row!!!
        //void operator()(T* ptr) const { cudaFree(ptr); }
        void operator()(T*) const { /* do nothing */ }
    };
    std::unique_ptr<T[],cuda_deleter> m_device_memory;
#endif

    memory(unsigned int size_, const T& value)
    : m_size{size_}
    , m_host_memory{ new T[m_size] }
    {
        for (unsigned int i=0; i<m_size; ++i)
            m_host_memory[i] = value;
#ifdef GHEX_CUDACC
        void* ptr;
        GHEX_CHECK_CUDA_RESULT(cudaMalloc(&ptr, m_size*sizeof(T)))
        m_device_memory.reset((T*)ptr);
        clone_to_device();
#endif
    }

    memory(const memory&) = delete;
    memory(memory&&) = default;

    T* data() const { return m_host_memory.get(); }
    T* host_data() const { return m_host_memory.get(); }
#ifdef GHEX_CUDACC
    T* device_data() const { return m_device_memory.get(); }
#endif

    unsigned int size() const { return m_size; }

    const T& operator[](unsigned int i) const { return m_host_memory[i]; }
          T& operator[](unsigned int i)       { return m_host_memory[i]; }

    T* begin() { return m_host_memory.get(); }
    T* end() { return m_host_memory.get()+m_size; }

    const T* begin() const { return m_host_memory.get(); }
    const T* end() const { return m_host_memory.get()+m_size; }

#ifdef GHEX_CUDACC
    void clone_to_device()
    {
        GHEX_CHECK_CUDA_RESULT(cudaMemcpy(m_device_memory.get(), m_host_memory.get(),
            m_size*sizeof(T), cudaMemcpyHostToDevice))
    }
    void clone_to_host()
    {
        GHEX_CHECK_CUDA_RESULT(cudaMemcpy(m_host_memory.get(),m_device_memory.get(),
            m_size*sizeof(T), cudaMemcpyDeviceToHost))
    }
#endif
};

memory<gridtools::array<int,2>> allocate_field()
{ 
    return {(HALO*2+DIM) * (HALO*2+DIM/2), gridtools::array<int,2>{-1,-1}};
}

template<typename RawField>
auto wrap_cpu_field(RawField& raw_field, const domain& d)
{
    return wrap_field<cpu,::gridtools::layout_map<1,0>>(d, raw_field.data(), arr{HALO, HALO}, arr{HALO*2+DIM, HALO*2+DIM/2});
}

#ifdef GHEX_CUDACC
template<typename RawField>
auto wrap_gpu_field(RawField& raw_field, const domain& d)
{
    return wrap_field<gpu,::gridtools::layout_map<1,0>>(d, raw_field.device_data(), arr{HALO, HALO}, arr{HALO*2+DIM, HALO*2+DIM/2});
}
#endif

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

template<typename Arr>
bool compare(const Arr& v, int x, int y)
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

struct domain_lu
{
    struct neighbor
    {
        int m_rank;
        int m_id;
        int rank() const noexcept { return m_rank; }
        int id() const noexcept { return m_id; }
    };
    
    arr m_dims;

    neighbor operator()(int id, arr const & offset) const noexcept
    {
        auto rank = id/2;
        auto y_ = id - 2*rank;
        auto y = rank/m_dims[0];
        auto x = rank - y*m_dims[0];
        y = ((2*y+y_ + offset[1]) + m_dims[1]*2)%(m_dims[1]*2);
        x = (x + offset[0] + m_dims[0])%m_dims[0];

        int n_rank = (y/2)*m_dims[0] + x;
        int n_id = 2*n_rank + (y%2);
        return {n_rank, n_id};
    }
};

template<typename Context, typename Pattern, typename SPattern, typename Domains, typename Barrier>
bool run(Context& context, const Pattern& pattern, const SPattern& spattern, const Domains& domains, const arr& dims, int thread_id, Barrier& barrier)
{
    bool res = true;
    // field
    auto raw_field = allocate_field();
    auto field     = fill(wrap_cpu_field(raw_field, domains[thread_id]));
#ifdef GHEX_CUDACC
    if (thread_id != 0)
        raw_field.clone_to_device();
    auto field_gpu = wrap_gpu_field(raw_field, domains[thread_id]);
#endif

    // get a communcator
    auto comm = context.get_communicator();
    
    // general exchange
    // ================
    auto co = make_communication_object<Pattern>(comm);

    // classical
    // ---------

#ifdef GHEX_CUDACC
    if (thread_id == 0)
        co.exchange(pattern(field)).wait();
    else
        co.exchange(pattern(field_gpu)).wait();
#else
    co.exchange(pattern(field)).wait();
#endif

    // check field
#ifdef GHEX_CUDACC
    if (thread_id != 0)
        raw_field.clone_to_host();
#endif
    res = res && check(field, dims);

    // reset field
    reset(field);
#ifdef GHEX_CUDACC
    if (thread_id != 0)
        raw_field.clone_to_device();
#endif

    barrier(comm);

    // using stages
    // ------------

#ifdef GHEX_CUDACC
    if (thread_id == 0)
    {
        co.exchange(spattern[0]->operator()(field)).wait();
        co.exchange(spattern[1]->operator()(field)).wait();
    }
    else
    {
        co.exchange(spattern[0]->operator(field_gpu)).wait();
        co.exchange(spattern[1]->operator(field_gpu)).wait();
    }
#else
    co.exchange(spattern[0]->operator()(field)).wait();
    co.exchange(spattern[1]->operator()(field)).wait();
#endif

    // check field
#ifdef GHEX_CUDACC
    if (thread_id != 0)
        raw_field.clone_to_host();
#endif
    res = res && check(field, dims);

    // reset field
    reset(field);
#ifdef GHEX_CUDACC
    if (thread_id != 0)
        raw_field.clone_to_device();
#endif

    barrier(comm);


    // bulk exchange (rma)
    // ===================

    // classical
    // ---------

#ifdef GHEX_CUDACC
    auto bco = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field), decltype(field_gpu)>(co);
    if (thread_id == 0)
        bco.add_field(pattern(field));
    else
        bco.add_field(pattern(field_gpu));
#else
    auto bco = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field)>(co);
    bco.add_field(pattern(field));
#endif
    //bco.exchange().wait();
    generic_bulk_communication_object gbco(std::move(bco));
    gbco.exchange().wait();

    // check field
#ifdef GHEX_CUDACC
    if (thread_id != 0)
        raw_field.clone_to_host();
#endif
    res = res && check(field, dims);

    // reset field
    reset(field);
#ifdef GHEX_CUDACC
    if (thread_id != 0)
        raw_field.clone_to_device();
#endif

    barrier(comm);

    // using stages
    // ------------

#ifdef GHEX_CUDACC
    auto bco_x = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field), decltype(field_gpu)>(co);
    auto bco_y = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field), decltype(field_gpu)>(co);
    if (thread_id == 0)
    {
        bco_x.add_field(spattern[0]->operator()(field));
        bco_y.add_field(spattern[1]->operator()(field));
    }
    else
    {
        bco_x.add_field(spattern[0]->operator()(field_gpu));
        bco_y.add_field(spattern[1]->operator()(field_gpu));
    }
#else
    auto bco_x = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field)>(co);
    auto bco_y = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field)>(co);
    bco_x.add_field(spattern[0]->operator()(field));
    bco_y.add_field(spattern[1]->operator()(field));
#endif
    //bco.exchange().wait();
    generic_bulk_communication_object gbco_x(std::move(bco_x));
    generic_bulk_communication_object gbco_y(std::move(bco_y));
    gbco_x.exchange().wait();
    gbco_y.exchange().wait();

    // check field
#ifdef GHEX_CUDACC
    if (thread_id != 0)
        raw_field.clone_to_host();
#endif
    res = res && check(field, dims);
    return res;
}

template<typename Context, typename Pattern, typename SPattern, typename Domains, typename Barrier>
bool run(Context& context, const Pattern& pattern, const SPattern& spattern, const Domains& domains, const arr& dims, Barrier& barrier)
{
    bool res = true;
    // fields
    auto raw_field_a = allocate_field();
    auto raw_field_b = allocate_field();
    auto field_a     = fill(wrap_cpu_field(raw_field_a, domains[0]));
    auto field_b     = fill(wrap_cpu_field(raw_field_b, domains[1]));
#ifdef GHEX_CUDACC
    raw_field_b.clone_to_device();
    auto field_b_gpu = wrap_gpu_field(raw_field_b, domains[1]);
#endif
    // get a communcator
    auto comm = context.get_communicator();
    
    // general exchange
    // ================
    auto co = make_communication_object<Pattern>(comm);

    // classical
    // ---------

#ifdef GHEX_CUDACC
    co.exchange(pattern(field_a), pattern(field_b_gpu)).wait();
#else
    co.exchange(pattern(field_a), pattern(field_b)).wait();
#endif

    // check fields
#ifdef GHEX_CUDACC
    raw_field_b.clone_to_host();
#endif
    res = res && check(field_a, dims);
    res = res && check(field_b, dims);

    // reset fields
    reset(field_a);
    reset(field_b);
#ifdef GHEX_CUDACC
    raw_field_b.clone_to_device();
#endif

    barrier(comm);

    // using stages
    // ------------

#ifdef GHEX_CUDACC
    co.exchange(spattern[0]->operator()(field_a), spattern[0]->operator()(field_b_gpu)).wait();
    co.exchange(spattern[1]->operator()(field_a), spattern[1]->operator()(field_b_gpu)).wait();
#else
    co.exchange(spattern[0]->operator()(field_a), spattern[0]->operator()(field_b)).wait();
    co.exchange(spattern[1]->operator()(field_a), spattern[1]->operator()(field_b)).wait();
#endif

    // check fields
#ifdef GHEX_CUDACC
    raw_field_b.clone_to_host();
#endif
    res = res && check(field_a, dims);
    res = res && check(field_b, dims);

    // reset fields
    reset(field_a);
    reset(field_b);
#ifdef GHEX_CUDACC
    raw_field_b.clone_to_device();
#endif

    barrier(comm);

    // bulk exchange (rma)
    // ===================
    
    // classical
    // ---------
    
#ifdef GHEX_CUDACC
    auto bco = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field_a), decltype(field_b_gpu)>(co);
    bco.add_field(pattern(field_a));
    bco.add_field(pattern(field_b_gpu));
#else
    auto bco = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field_a)>(co);
    bco.add_field(pattern(field_a));
    bco.add_field(pattern(field_b));
#endif
    bco.init();
    barrier(comm);
    bco.exchange().wait();

    // check fields
#ifdef GHEX_CUDACC
    raw_field_b.clone_to_host();
#endif
    res = res && check(field_a, dims);
    res = res && check(field_b, dims);

    // reset fields
    reset(field_a);
    reset(field_b);
#ifdef GHEX_CUDACC
    raw_field_b.clone_to_device();
#endif

    barrier(comm);

    // using stages
    // ------------

#ifdef GHEX_CUDACC
    auto bco_x = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field_a), decltype(field_b_gpu)>(co);
    bco_x.add_field(spattern[0]->operator()(field_a));
    bco_x.add_field(spattern[0]->operator()(field_b_gpu));
    auto bco_y = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field_a), decltype(field_b_gpu)>(co);
    bco_y.add_field(spattern[1]->operator()(field_a));
    bco_y.add_field(spattern[1]->operator()(field_b_gpu));
#else
    auto bco_x = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field_a)>(co,true);
    bco_x.add_field(spattern[0]->operator()(field_a));
    bco_x.add_field(spattern[0]->operator()(field_b));
    auto bco_y = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field_a)>(co,true);
    bco_y.add_field(spattern[1]->operator()(field_a));
    bco_y.add_field(spattern[1]->operator()(field_b));
#endif
    bco_x.init();
    bco_y.init();
    barrier(comm);
    bco_x.exchange().wait();
    bco_y.exchange().wait();

    // check fields
#ifdef GHEX_CUDACC
    raw_field_b.clone_to_host();
#endif
    res = res && check(field_a, dims);
    res = res && check(field_b, dims);

    return res;
}

void sim(bool multi_threaded)
{
    // make a context from mpi world and number of threads
    auto context_ptr = factory::create(MPI_COMM_WORLD);
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
    // neighbor lookup
    domain_lu d_lu{dims};
    //if (context.rank()==2)
    //{
    //    {
    //    auto n_0 = d_lu(context.rank()*2+1, {0,-1});
    //    std::cout << n_0.rank() << ", " << n_0.id() << std::endl;
    //    std::cout << domains[0].domain_id() << std::endl;
    //    }
    //    {
    //    auto n_0 = d_lu(context.rank()*2, {0,1});
    //    std::cout << n_0.rank() << ", " << n_0.id() << std::endl;
    //    std::cout << domains[1].domain_id() << std::endl;
    //    }
    //}

    auto staged_pattern = structured::regular::make_staged_pattern(context, domains, d_lu,
        arr{0,0}, arr{dims[0]*DIM-1,dims[1]*DIM-1}, halos, periodic);

    //if (context.rank()==1)
    //{
    //int i = 0;
    //for (auto& p_cont_ptr : staged_pattern)
    //{
    //    std::cout << "i = " << i << std::endl;
    //    ++i;
    //    for (auto& p : *p_cont_ptr)
    //    {
    //        std::cout << "pattern domain id = " << p.domain_id() << std::endl;
    //        std::cout << "recv halos: " << std::endl;
    //        for (auto& r : p.recv_halos())
    //        {
    //            std::cout << r.first << ": " << std::endl;
    //            for (auto& h : r.second)
    //                std::cout << "  " << h << std::endl;
    //        }
    //        std::cout << "send halos: " << std::endl;
    //        for (auto& r : p.send_halos())
    //        {
    //            std::cout << r.first << ": " << std::endl;
    //            for (auto& h : r.second)
    //                std::cout << "  " << h << std::endl;
    //        }
    //        std::cout << std::endl;
    //    }
    //}
    //}

    // make halo generator
    halo_gen gen{arr{0,0}, arr{dims[0]*DIM-1,dims[1]*DIM-1}, halos, periodic};
    // create a pattern for communication
    auto pattern = make_pattern<structured::grid>(context, gen, domains);
    // run
    bool res = true;
    if (multi_threaded)
    {
        tl::barrier_t barrier(2);
        auto run_fct = [&context,&pattern,&staged_pattern,&domains,&dims,&barrier](int id)
            { return run(context, pattern, staged_pattern, domains, dims, id, barrier); };
        auto f1 = std::async(std::launch::async, run_fct, 0);
        auto f2 = std::async(std::launch::async, run_fct, 1);
        res = res && f1.get();
        res = res && f2.get();
    }
    else
    {
        tl::barrier_t barrier(1);
        res = res && run(context, pattern, staged_pattern, domains, dims, barrier);
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
    sim(true);
}

