

#include "../include/ghex/regular_domain.hpp"
#include "../include/ghex/communication_object.hpp"
#include "../include/ghex/exchange.hpp"

#include "gtest/gtest.h"
#include <fstream>
#include <gridtools/common/layout_map.hpp>
//#include <gridtools/common/boollist.hpp>
//#include <gridtools/communication/halo_exchange.hpp>
//#include <gridtools/storage/storage_facility.hpp>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <sys/time.h>

#include "triplet.hpp"
//#include <gridtools/regression/communtriplet.hpp>

//#include <gridtools/tools/mpi_unit_test_driver/device_binding.hpp>

#define STANDALONE


namespace halo_exchange_3D_generic_full {
    int pid;
    int nprocs;
    MPI_Comm CartComm;
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};

    struct timeval start_tv;
    struct timeval stop1_tv;
    struct timeval stop2_tv;
    struct timeval stop3_tv;
    double lapse_time1;
    double lapse_time2;
    double lapse_time3;
    double lapse_time4;

#define B_ADD 1
#define C_ADD 2

    typedef int T1;
    typedef double T2;
    typedef long long int T3;

    template<typename T, int I1, int I2, int I3>
    using array_t = array<triple_t<USE_DOUBLE, T>, gridtools::layout_map<I1,I2,I3>>;


    //template<typename T, int I1, int I2, int I3>
    struct field_descriptor
    {
        //using array_type = array_t<T,I1,I2,I3>;
        using index_type = int;
        using global_index_type = int;
        using domain_index_type = int;

        //array_type m_impl;
        int hm[3];
        int hp[3];
        int dims[3];
        int origin[3];
        int extent[3];
        int domain_extent[3];
        int g_nx;
        int g_ny;
        int g_nz;
        bool periodic[3];

        field_descriptor(
            int DIM1,
            int DIM2,
            int DIM3,
            int H1m,
            int H1p,
            int H2m,
            int H2p,
            int H3m,
            int H3p,
            bool periodic1=true, 
            bool periodic2=true,
            bool periodic3=true)
            //triple_t<USE_DOUBLE, T> *_a ) :
            //m_impl(_a,DIM1+H1m+H1p,DIM2+H2m+H2p,DIM3+H3m+H3p)
        {
            periodic[0] = periodic1;
            periodic[1] = periodic2;
            periodic[2] = periodic3;
            hm[0] = H1m;
            hm[1] = H2m;
            hm[2] = H3m;
            hp[0] = H1p;
            hp[1] = H2p;
            hp[2] = H3p;
            extent[0] = DIM1+H1m+H1p;
            extent[1] = DIM2+H2m+H2p;
            extent[2] = DIM3+H3m+H3p;
            domain_extent[0] = DIM1;
            domain_extent[1] = DIM2;
            domain_extent[2] = DIM3;
            int periods[3];
            int coords[3];
            MPI_Cart_get(CartComm, 3, dims, periods, coords);
            g_nx = dims[0]*DIM1;
            g_ny = dims[1]*DIM2;
            g_nz = dims[2]*DIM3;
            origin[0] = coords[0]*DIM1;
            origin[1] = coords[1]*DIM2;
            origin[2] = coords[2]*DIM3;
        }

        global_index_type global_index(index_type i, index_type j, index_type k) const noexcept
        {
            if (!periodic[0] && ((i-hm[0]+origin[0])<0 || (i-hm[0]+origin[0])>=g_nx)) return -1;
            if (!periodic[1] && ((j-hm[1]+origin[1])<0 || (j-hm[1]+origin[1])>=g_ny)) return -1;
            if (!periodic[2] && ((k-hm[2]+origin[2])<0 || (k-hm[2]+origin[2])>=g_nz)) return -1;
            return
              (((i-hm[0]+origin[0])+g_nx)%g_nx)
            + (((j-hm[1]+origin[1])+g_ny)%g_ny)*g_nx
            + (((k-hm[2]+origin[2])+g_nz)%g_nz)*g_nx*g_ny;
        }

        domain_index_type domain_id(global_index_type gid) const noexcept
        {
            if (gid == -1) return -1;
            const auto k = gid/(g_nx*g_ny);
            gid -= k*(g_nx*g_ny);
            const auto j = gid/(g_nx);
            gid -= j*(g_nx);
            const auto i = gid;
            const auto xx = i/domain_extent[0];
            const auto yy = j/domain_extent[1];
            const auto zz = k/domain_extent[2];
            return xx+yy*dims[0]+zz*dims[0]*dims[1];
        }

        int rank(domain_index_type did) const noexcept
        {
            int coords[3];
            coords[2] = did/(dims[0]*dims[1]);
            did -= coords[2]*(dims[0]*dims[1]);
            coords[1] = did/(dims[0]);
            did -= coords[1]*dims[0];
            coords[0] = did;
            int r;
            MPI_Cart_rank(CartComm, coords, &r);
            return r;
        }
    };



}

namespace ghex {

template<>
struct domain_id_traits<int>
{
    static constexpr int invalid = -1;
};

}

/*namespace ghex {

    template<typename T, int I1, int I2, int I3>
    struct data_field_traits<halo_exchange_3D_generic_full::array_t<T,I1,I2,I3>>
    {
        using data_field_type        = halo_exchange_3D_generic_full::array_t<T,I1,I2,I3>; 
        using value_type             = T;
        using local_cell_index_type  = std::tuple<int,int,int>;
        using global_cell_index_type = int;
    };

    template<typename T, int I1, int I2, int I3>
    struct regular_data_field_traits<halo_exchange_3D_generic_full::array_t<T,I1,I2,I3>>
    {
        using data_field_type        = halo_exchange_3D_generic_full::array_t<T,I1,I2,I3>; 
        using value_type             = T;
        //using local_cell_index_type  = std::tuple<int,int,int>;
        using index_type             = int; 
        using global_cell_index_type = int;
        using dimension              = std::integral_constant<int,3>;

        //global_cell_index(daint i, int j, int k) const 
    };
}*/

namespace halo_exchange_3D_generic_full {

    template <typename ST, int I1, int I2, int I3, bool per0, bool per1, bool per2>
    bool run(ST &file,
        int DIM1,
        int DIM2,
        int DIM3,
        int H1m1,
        int H1p1,
        int H2m1,
        int H2p1,
        int H3m1,
        int H3p1,
        int H1m2,
        int H1p2,
        int H2m2,
        int H2p2,
        int H3m2,
        int H3p2,
        int H1m3,
        int H1p3,
        int H2m3,
        int H2p3,
        int H3m3,
        int H3p3,
        triple_t<USE_DOUBLE, T1> *_a,
        triple_t<USE_DOUBLE, T2> *_b,
        triple_t<USE_DOUBLE, T3> *_c) 
    {

        typedef gridtools::layout_map<I1, I2, I3> layoutmap;

        array<triple_t<USE_DOUBLE, T1>, layoutmap> a(
            _a, (DIM1 + H1m1 + H1p1), (DIM2 + H2m1 + H2p1), (DIM3 + H3m1 + H3p1));
        array<triple_t<USE_DOUBLE, T2>, layoutmap> b(
            _b, (DIM1 + H1m2 + H1p2), (DIM2 + H2m2 + H2p2), (DIM3 + H3m2 + H3p2));
        array<triple_t<USE_DOUBLE, T3>, layoutmap> c(
            _c, (DIM1 + H1m3 + H1p3), (DIM2 + H2m3 + H2p3), (DIM3 + H3m3 + H3p3));


        file << "Proc: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")\n";

        /* Just an initialization */
        for (int ii = 0; ii < DIM1 + H1m1 + H1p1; ++ii)
            for (int jj = 0; jj < DIM2 + H2m1 + H2p1; ++jj) {
                for (int kk = 0; kk < DIM3 + H3m1 + H3p1; ++kk) {
                    a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>();
                }
            }

        for (int ii = 0; ii < DIM1 + H1m2 + H1p2; ++ii)
            for (int jj = 0; jj < DIM2 + H2m2 + H2p2; ++jj) {
                for (int kk = 0; kk < DIM3 + H3m2 + H3p2; ++kk) {
                    b(ii, jj, kk) = triple_t<USE_DOUBLE, T2>();
                }
            }

        for (int ii = 0; ii < DIM1 + H1m3 + H1p3; ++ii)
            for (int jj = 0; jj < DIM2 + H2m3 + H2p3; ++jj) {
                for (int kk = 0; kk < DIM3 + H3m3 + H3p3; ++kk) {
                    c(ii, jj, kk) = triple_t<USE_DOUBLE, T3>();
                }
            }

        for (int ii = H1m1; ii < DIM1 + H1m1; ++ii)
            for (int jj = H2m1; jj < DIM2 + H2m1; ++jj)
                for (int kk = H3m1; kk < DIM3 + H3m1; ++kk) {
                    a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>(
                        ii - H1m1 + (DIM1)*coords[0], jj - H2m1 + (DIM2)*coords[1], kk - H3m1 + (DIM3)*coords[2]);
                }

        for (int ii = H1m2; ii < DIM1 + H1m2; ++ii)
            for (int jj = H2m2; jj < DIM2 + H2m2; ++jj)
                for (int kk = H3m2; kk < DIM3 + H3m2; ++kk) {
                    b(ii, jj, kk) = triple_t<USE_DOUBLE, T2>(ii - H1m2 + (DIM1)*coords[0] + B_ADD,
                        jj - H2m2 + (DIM2)*coords[1] + B_ADD,
                        kk - H3m2 + (DIM3)*coords[2] + B_ADD);
                }

        for (int ii = H1m3; ii < DIM1 + H1m3; ++ii)
            for (int jj = H2m3; jj < DIM2 + H2m3; ++jj)
                for (int kk = H3m3; kk < DIM3 + H3m3; ++kk) {
                    c(ii, jj, kk) = triple_t<USE_DOUBLE, T3>(ii - H1m3 + (DIM1)*coords[0] + C_ADD,
                        jj - H2m3 + (DIM2)*coords[1] + C_ADD,
                        kk - H3m3 + (DIM3)*coords[2] + C_ADD);
                }

        file << "A \n";
        printbuff(file, a, DIM1 + H1m1 + H1p1, DIM2 + H2m1 + H2p1, DIM3 + H3m1 + H3p1);
        file << "B \n";
        printbuff(file, b, DIM1 + H1m2 + H1p2, DIM2 + H2m2 + H2p2, DIM3 + H3m2 + H3p2);
        file << "C \n";
        printbuff(file, c, DIM1 + H1m3 + H1p3, DIM2 + H2m3 + H2p3, DIM3 + H3m3 + H3p3);
        file.flush();


        field_descriptor a_desc(DIM1,DIM2,DIM3,H1m1,H1p1,H2m1,H2p1,H3m1,H3p1, per0, per1, per2);
        field_descriptor b_desc(DIM1,DIM2,DIM3,H1m2,H1p2,H2m2,H2p2,H3m2,H3p2, per0, per1, per2);
        field_descriptor c_desc(DIM1,DIM2,DIM3,H1m3,H1p3,H2m3,H2p3,H3m3,H3p3, per0, per1, per2);

        using regular_domain_t = ghex::regular_domain<3,int,int,int>;


        regular_domain_t a_domain(
            std::array<int,3>{H1m1,H2m1,H3m1}, // origin
            std::array<int,3>{DIM1,DIM2,DIM3},
            std::array<int,3>{H1m1,H2m1,H3m1},
            std::array<int,3>{H1p1,H2p1,H3p1},
            [&a_desc](int i, int j, int k) { return a_desc.global_index(i,j,k); },
            [&a_desc](int i) { return a_desc.domain_id(i); }
        );
        regular_domain_t b_domain(
            std::array<int,3>{H1m2,H2m2,H3m2},
            std::array<int,3>{DIM1,DIM2,DIM3},
            std::array<int,3>{H1m2,H2m2,H3m2},
            std::array<int,3>{H1p2,H2p2,H3p2},
            [&b_desc](int i, int j, int k) { return b_desc.global_index(i,j,k); },
            [&b_desc](int i) { return b_desc.domain_id(i); }
        );
        regular_domain_t c_domain(
            std::array<int,3>{H1m3,H2m3,H3m3},
            std::array<int,3>{DIM1,DIM2,DIM3},
            std::array<int,3>{H1m3,H2m3,H3m3},
            std::array<int,3>{H1p3,H2p3,H3p3},
            [&c_desc](int i, int j, int k) { return c_desc.global_index(i,j,k); },
            [&c_desc](int i) { return c_desc.domain_id(i); }
        );

        using co_a_t = ghex::communication_object<regular_domain_t, triple_t<USE_DOUBLE, T1>, ghex::protocol::mpi_async>;
        using co_b_t = ghex::communication_object<regular_domain_t, triple_t<USE_DOUBLE, T2>, ghex::protocol::mpi_async>;
        using co_c_t = ghex::communication_object<regular_domain_t, triple_t<USE_DOUBLE, T3>, ghex::protocol::mpi_async>;

        co_a_t co_a(a_domain, [&a_desc](int i) { return a_desc.rank(i);});
        co_b_t co_b(b_domain, [&b_desc](int i) { return b_desc.rank(i);});
        co_c_t co_c(c_domain, [&c_desc](int i) { return c_desc.rank(i);});

        using exchange_t = ghex::exchange<std::allocator<double>, co_a_t, co_b_t, co_c_t>;

        exchange_t ex(CartComm, co_a, co_b, co_c);

        MPI_Barrier(MPI_COMM_WORLD);

        gettimeofday(&start_tv, nullptr);

        ex.pack(a,b,c);
        
        gettimeofday(&stop1_tv, nullptr);
        
        ex.post();
        ex.wait();

        gettimeofday(&stop2_tv, nullptr);
        
        ex.unpack(a,b,c);

        gettimeofday(&stop3_tv, nullptr);
        

        lapse_time1 =
            ((static_cast<double>(stop1_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop1_tv.tv_usec)) -
                (static_cast<double>(start_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(start_tv.tv_usec))) *
            1000.0;

        lapse_time2 =
            ((static_cast<double>(stop2_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop2_tv.tv_usec)) -
                (static_cast<double>(stop1_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop1_tv.tv_usec))) *
            1000.0;

        lapse_time3 =
            ((static_cast<double>(stop3_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop3_tv.tv_usec)) -
                (static_cast<double>(stop2_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop2_tv.tv_usec))) *
            1000.0;

        lapse_time4 =
            ((static_cast<double>(stop3_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop3_tv.tv_usec)) -
                (static_cast<double>(start_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(start_tv.tv_usec))) *
            1000.0;

        MPI_Barrier(MPI_COMM_WORLD);
        file << "TIME PACK: " << lapse_time1 << std::endl;
        file << "TIME EXCH: " << lapse_time2 << std::endl;
        file << "TIME UNPK: " << lapse_time3 << std::endl;
        file << "TIME ALL : " << lapse_time1 + lapse_time2 + lapse_time3 << std::endl;
        file << "TIME TOT : " << lapse_time4 << std::endl;


        file << "\n********************************************************************************\n";

        file << "A \n";
        printbuff(file, a, DIM1 + H1m1 + H1p1, DIM2 + H2m1 + H2p1, DIM3 + H3m1 + H3p1);
        file << "B \n";
        printbuff(file, b, DIM1 + H1m2 + H1p2, DIM2 + H2m2 + H2p2, DIM3 + H3m2 + H3p2);
        file << "C \n";
        printbuff(file, c, DIM1 + H1m3 + H1p3, DIM2 + H2m3 + H2p3, DIM3 + H3m3 + H3p3);
        file.flush();

        int passed = true;

        /* Checking the data arrived correctly in the whole region
         */
        for (int ii = 0; ii < DIM1 + H1m1 + H1p1; ++ii)
            for (int jj = 0; jj < DIM2 + H2m1 + H2p1; ++jj)
                for (int kk = 0; kk < DIM3 + H3m1 + H3p1; ++kk) {

                    triple_t<USE_DOUBLE, T1> ta;
                    int tax, tay, taz;

                    tax = modulus(ii - H1m1 + (DIM1)*coords[0], DIM1 * dims[0]);

                    tay = modulus(jj - H2m1 + (DIM2)*coords[1], DIM2 * dims[1]);

                    taz = modulus(kk - H3m1 + (DIM3)*coords[2], DIM3 * dims[2]);

                    if (!per0) {
                        if (((coords[0] == 0) && (ii < H1m1)) || ((coords[0] == dims[0] - 1) && (ii >= DIM1 + H1m1))) {
                            tax = triple_t<USE_DOUBLE, T1>().x();
                        }
                    }

                    if (!per1) {
                        if (((coords[1] == 0) && (jj < H2m1)) || ((coords[1] == dims[1] - 1) && (jj >= DIM2 + H2m1))) {
                            tay = triple_t<USE_DOUBLE, T1>().y();
                        }
                    }

                    if (!per2) {
                        if (((coords[2] == 0) && (kk < H3m1)) || ((coords[2] == dims[2] - 1) && (kk >= DIM3 + H3m1))) {
                            taz = triple_t<USE_DOUBLE, T1>().z();
                        }
                    }

                    ta = triple_t<USE_DOUBLE, T1>(tax, tay, taz).floor();

                    if (a(ii, jj, kk) != ta) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "a " << a(ii, jj, kk) << " != " << ta << "\n";
                    }
                }

        for (int ii = 0; ii < DIM1 + H1m2 + H1p2; ++ii)
            for (int jj = 0; jj < DIM2 + H2m2 + H2p2; ++jj)
                for (int kk = 0; kk < DIM3 + H3m2 + H3p2; ++kk) {

                    triple_t<USE_DOUBLE, T2> tb;
                    int tbx, tby, tbz;

                    tbx = modulus(ii - H1m2 + (DIM1)*coords[0], DIM1 * dims[0]) + B_ADD;

                    tby = modulus(jj - H2m2 + (DIM2)*coords[1], DIM2 * dims[1]) + B_ADD;

                    tbz = modulus(kk - H3m2 + (DIM3)*coords[2], DIM3 * dims[2]) + B_ADD;

                    if (!per0) {
                        if (((coords[0] == 0) && (ii < H1m2)) || ((coords[0] == dims[0] - 1) && (ii >= DIM1 + H1m2))) {
                            tbx = triple_t<USE_DOUBLE, T2>().x();
                        }
                    }

                    if (!per1) {
                        if (((coords[1] == 0) && (jj < H2m2)) || ((coords[1] == dims[1] - 1) && (jj >= DIM2 + H2m2))) {
                            tby = triple_t<USE_DOUBLE, T2>().y();
                        }
                    }

                    if (!per2) {
                        if (((coords[2] == 0) && (kk < H3m2)) || ((coords[2] == dims[2] - 1) && (kk >= DIM3 + H3m2))) {
                            tbz = triple_t<USE_DOUBLE, T2>().z();
                        }
                    }

                    tb = triple_t<USE_DOUBLE, T2>(tbx, tby, tbz).floor();

                    if (b(ii, jj, kk) != tb) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "b " << b(ii, jj, kk) << " != " << tb << "\n";
                    }
                }

        for (int ii = 0; ii < DIM1 + H1m3 + H1p3; ++ii)
            for (int jj = 0; jj < DIM2 + H2m3 + H2p3; ++jj)
                for (int kk = 0; kk < DIM3 + H3m3 + H3p3; ++kk) {

                    triple_t<USE_DOUBLE, T3> tc;
                    int tcx, tcy, tcz;

                    tcx = modulus(ii - H1m3 + (DIM1)*coords[0], DIM1 * dims[0]) + C_ADD;

                    tcy = modulus(jj - H2m3 + (DIM2)*coords[1], DIM2 * dims[1]) + C_ADD;

                    tcz = modulus(kk - H3m3 + (DIM3)*coords[2], DIM3 * dims[2]) + C_ADD;

                    if (!per0) {
                        if (((coords[0] == 0) && (ii < H1m3)) || ((coords[0] == dims[0] - 1) && (ii >= DIM1 + H1m3))) {
                            tcx = triple_t<USE_DOUBLE, T3>().x();
                        }
                    }

                    if (!per1) {
                        if (((coords[1] == 0) && (jj < H2m3)) || ((coords[1] == dims[1] - 1) && (jj >= DIM2 + H2m3))) {
                            tcy = triple_t<USE_DOUBLE, T3>().y();
                        }
                    }

                    if (!per2) {
                        if (((coords[2] == 0) && (kk < H3m3)) || ((coords[2] == dims[2] - 1) && (kk >= DIM3 + H3m3))) {
                            tcz = triple_t<USE_DOUBLE, T3>().z();
                        }
                    }

                    tc = triple_t<USE_DOUBLE, T3>(tcx, tcy, tcz).floor();

                    if (c(ii, jj, kk) != tc) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "c " << c(ii, jj, kk) << " != " << tc << "\n";
                    }
                }

        if (passed)
            file << "RESULT: PASSED!\n";
        else
            file << "RESULT: FAILED!\n";

        return passed;
    }

    bool test(int DIM1,
        int DIM2,
        int DIM3,
        int H1m1,
        int H1p1,
        int H2m1,
        int H2p1,
        int H3m1,
        int H3p1,
        int H1m2,
        int H1p2,
        int H2m2,
        int H2p2,
        int H3m2,
        int H3p2,
        int H1m3,
        int H1p3,
        int H2m3,
        int H2p3,
        int H3m3,
        int H3p3) {

        /* Here we compute the computing gris as in many applications
         */
        MPI_Comm_rank(MPI_COMM_WORLD, &pid);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

        std::cout << pid << " " << nprocs << "\n";

        std::stringstream ss;
        ss << pid;

        std::string filename = "out" + ss.str() + ".txt";

        std::cout << filename << std::endl;
        std::ofstream file(filename.c_str());

        file << pid << "  " << nprocs << "\n";

        MPI_Dims_create(nprocs, 3, dims);
        int period[3] = {1, 1, 1};

        file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";

        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

        MPI_Cart_get(CartComm, 3, dims, period, coords);

        /* Each process will hold a tile of size
           (DIM1+2*H)x(DIM2+2*H)x(DIM3+2*H). The DIM1xDIM2xDIM3 area inside
           the H width border is the inner region of an hypothetical stencil
           computation whose halo width is H.
        */

        file << "Field A "
             << "size = " << DIM1 << "x" << DIM2 << "x" << DIM3 << " "
             << "Halo along i " << H1m1 << " - " << H1p1 << ", "
             << "Halo along j " << H2m1 << " - " << H2p1 << ", "
             << "Halo along k " << H3m1 << " - " << H3p1 << std::endl;

        file << "Field B "
             << "size = " << DIM1 << "x" << DIM2 << "x" << DIM3 << " "
             << "Halo along i " << H1m2 << " - " << H1p2 << ", "
             << "Halo along j " << H2m2 << " - " << H2p2 << ", "
             << "Halo along k " << H3m2 << " - " << H3p2 << std::endl;

        file << "Field C "
             << "size = " << DIM1 << "x" << DIM2 << "x" << DIM3 << " "
             << "Halo along i " << H1m3 << " - " << H1p3 << ", "
             << "Halo along j " << H2m3 << " - " << H2p3 << ", "
             << "Halo along k " << H3m3 << " - " << H3p3 << std::endl;
        file.flush();

        /* This example will exchange 3 data arrays at the same time with
           different values.
        */
        triple_t<USE_DOUBLE, T1> *_a =
            new triple_t<USE_DOUBLE, T1>[(DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1)];
        triple_t<USE_DOUBLE, T2> *_b =
            new triple_t<USE_DOUBLE, T2>[(DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2)];
        triple_t<USE_DOUBLE, T3> *_c =
            new triple_t<USE_DOUBLE, T3>[(DIM1 + H1m3 + H1p3) * (DIM2 + H2m3 + H2p3) * (DIM3 + H3m3 + H3p3)];
        
        bool passed = true;

        file << "Permutation 0,1,2\n";

        file << "run<std::ostream, 0,1,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        
        passed = passed and run<std::ostream, 0, 1, 2, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,1,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,1,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        
        file << "run<std::ostream, 0,1,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        
        file << "run<std::ostream, 0,1,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        
        file
            << "run<std::ostream, 0,1,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 0,1,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,1,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 0,2,1\n";

        file << "run<std::ostream, 0,2,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,2,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,2,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 0,2,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,2,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 0,2,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 0,2,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,2,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 1,0,2\n";

        file << "run<std::ostream, 1,0,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,0,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,0,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,0,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,0,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,0,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,0,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,0,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 1,2,0\n";

        file << "run<std::ostream, 1,2,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,2,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,2,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,2,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,2,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,2,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,2,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,2,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H31, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 2,0,1\n";

        file << "run<std::ostream, 2,0,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,0,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,0,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,0,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,0,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,0,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,0,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,0,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 2,1,0\n";

        file << "run<std::ostream, 2,1,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,1,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,1,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,1,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,1,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,1,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,1,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,1,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";
        
        delete[] _a;
        delete[] _b;
        delete[] _c;

        MPI_Comm_free(&CartComm);
        return passed;
    }

} // namespace halo_exchange_3D_generic_full

#ifdef STANDALONE
int main(int argc, char **argv) {

    int p;
    MPI_Init(&argc, &argv);
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &p);
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &p);
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);
    //gridtools::GCL_Init(argc, argv);

    if (argc != 22) {
        std::cout << "Usage: test_halo_exchange_3D dimx dimy dimz h1m1 hip1 h2m1 h2m1 h3m1 h3p1 h1m2 hip2 h2m2 h2m2 "
                     "h3m2 h3p2 h1m3 hip3 h2m3 h2m3 h3m3 h3p3\n where args are integer sizes of the data fields and "
                     "halo width"
                  << std::endl;
        return 1;
    }
    int DIM1 = atoi(argv[1]);
    int DIM2 = atoi(argv[2]);
    int DIM3 = atoi(argv[3]);
    int H1m1 = atoi(argv[4]);
    int H1p1 = atoi(argv[5]);
    int H2m1 = atoi(argv[6]);
    int H2p1 = atoi(argv[7]);
    int H3m1 = atoi(argv[8]);
    int H3p1 = atoi(argv[9]);
    int H1m2 = atoi(argv[10]);
    int H1p2 = atoi(argv[11]);
    int H2m2 = atoi(argv[12]);
    int H2p2 = atoi(argv[13]);
    int H3m2 = atoi(argv[14]);
    int H3p2 = atoi(argv[15]);
    int H1m3 = atoi(argv[16]);
    int H1p3 = atoi(argv[17]);
    int H2m3 = atoi(argv[18]);
    int H2p3 = atoi(argv[19]);
    int H3m3 = atoi(argv[20]);
    int H3p3 = atoi(argv[21]);

    halo_exchange_3D_generic_full::test(DIM1,
        DIM2,
        DIM3,
        H1m1,
        H1p1,
        H2m1,
        H2p1,
        H3m1,
        H3p1,
        H1m2,
        H1p2,
        H2m2,
        H2p2,
        H3m2,
        H3p2,
        H1m3,
        H1p3,
        H2m3,
        H2p3,
        H3m3,
        H3p3);

    MPI_Finalize();
    return 0;
}
#else
/*TEST(Communication, test_halo_exchange_3D_generic_full) {
    bool passed = halo_exchange_3D_generic_full::test(98, 54, 87, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 0, 1);
    EXPECT_TRUE(passed);*/
}
#endif

