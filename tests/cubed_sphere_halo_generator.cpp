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

#include <ghex/structured/pattern.hpp>
#include <ghex/communication_object_2.hpp>
#include <ghex/threads/none/primitives.hpp>
#include <ghex/cubed_sphere/halo_generator.hpp>
#include <ghex/cubed_sphere/field_descriptor.hpp>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#else
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#endif
using threading = gridtools::ghex::threads::none::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;

#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>

// cubed sphere tiles and coordinate system
//
//            +----------+
//            |       (2)|
//            |      x ∧ |
//            |        | |
//            |    y<--+ |
// +----------+----------+----------+----------+
// | +-->y (4)|       (0)|       (1)| +-->y (3)|
// | |        | ∧ y      | ∧ y      | |        |
// | ∨ x      | |        | |        | ∨ x      |
// |          | +-->x    | +-->x    |          |
// +----------+----------+----------+----------+
//            |       (5)|
//            | ∧ y      |
//            | |        |
//            | +-->x    |
//            +----------+
//
// here we have each tile/face decomposed into 4 parts
// neigborhood of a tile/face is depicted below
//              
//+-------------------------------------------------------+-------------------------------------------------------+
//:                      even tiles                       :                      odd tiles                        :
//+-------------------------------------------------------+-------------------------------------------------------+
//:                  +--------+--------+                  :                  +--------+--------+                  :
//:                  |        |        |                  :                  |        |        |                  :
//:                  |        |        |                  :                  |        |        |                  :
//:                  |       3|1       |                  :                  |       2|3       |                  :
//:                  +--------+--------+                  :                  +--------+--------+                  :
//:                  |       2|0   x ∧ |                  :                  | ∧ y   0|1       |                  :
//:                  |        |      | |                  :                  | |      |        |                  :
//:                  |        |  y<--+ |                  :                  | +-->x  |        |                  :
//:+--------+--------+--------+--------+--------+--------+:+--------+--------+--------+--------+--------+--------+:
//:| +-->y  |        |        |        |        |        |:|        |        |        |        | +-->y  |        |:
//:| |      |        |        |        |        |        |:|        |        |        |        | |      |        |:
//:| ∨ x   0|2       |       2|3       |       2|3       |:|       2|3       |       2|3       | ∨ x   0|3       |:
//:+--------+--------+--------+--------+--------+--------+:+--------+--------+--------+--------+--------+--------+:
//:|       1|3       | ∧ y   0|1       | ∧ y   0|1       |:| ∧ y   0|1       | ∧ y   0|1       |       1|2       |:
//:|        |        | |      |        | |      |        |:| |      |        | |      |        |        |        |:
//:|        |        | +-->x  |        | +-->x  |        |:| +-->x  |        | +-->x  |        |        |        |:
//:+--------+--------+--------+--------+--------+--------+:+--------+--------+--------+--------+--------+--------+:
//:                  |        |        |                  :                  |        |        |                  :
//:                  |        |        |                  :                  |        |        |                  :
//:                  |       2|3       |                  :                  |       3|1       |                  :
//:                  +--------+--------+                  :                  +--------+--------+                  :
//:                  | ∧ y   0|1       |                  :                  |       2|0   x ∧ |                  :
//:                  | |      |        |                  :                  |        |      | |                  :
//:                  | +-->x  |        |                  :                  |        |  y<--+ |                  :
//:                  +--------+--------+                  :                  +--------+--------+                  :
//+-------------------------------------------------------+-------------------------------------------------------+

#define GHEX_CS_CHECK_HEADER                      \
    const auto x_dom_min = field.offsets()[1];    \
    const auto x_min     = x_dom_min-halo;        \
    const auto y_dom_min = field.offsets()[2];    \
    const auto y_min     = y_dom_min-halo;        \
    const auto x_dom_max = x_dom_min + n;         \
    const auto x_max     = x_dom_max+halo;        \
    const auto y_dom_max = y_dom_min + n;         \
    const auto y_max     = y_dom_max+halo;        \
    const auto strides   = field.byte_strides();  \
    using value_type = typename Field::value_type;

#define GHEX_CS_CHECK_VALUE                                                                \
    const auto memory_location = strides[0]*c + strides[1]*x + strides[2]*y+ strides[3]*z; \
    const value_type value = *reinterpret_cast<const value_type*>(                         \
        reinterpret_cast<const char*>(field.data())+memory_location);

// even checks
// -----------

template<typename Field>
void check_even_0(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[3]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect empty
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected = -1;
                    EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from neighbor tile -y id 2
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*2 + 1000*c + z +
                        100*(x-field.offsets()[1]) + 
                        10*(n+y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect from neighbor tile -y id 3
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(n+y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from neighbor tile -x id 3
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*3 + 1000*c + z +
                        100*(n-(y-field.offsets()[2])-1) +
                        10*(n+x-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check right         - expect from same tile id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from neighbor tile -x id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*2 + 1000*c + z +
                        100*(n-(y-field.offsets()[2]-n)-1) +
                        10*(n+x-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from same tile id 2
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                        100*(x-field.offsets()[1]) + 
                        10*(y-field.offsets()[2]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top right     - expect from same tile id 3
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(y-field.offsets()[2]-n);
                    EXPECT_EQ(value, expected);
                }
            }
        }
}

template<typename Field>
void check_even_1(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[3]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect from neighbor tile -y id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*2 + 1000*c + z +
                        100*(n+x-field.offsets()[1]) + 
                        10*(n+y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from neighbor tile -y id 3
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[1]) + 
                        10*(n+y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect empty
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected = -1;
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from same tile id 0
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*0 + 1000*c + z +
                        100*(n+x-field.offsets()[1]) +
                        10*(y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
                // check right         - expect from neighbor tile +x id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*0 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from same tile id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                        100*(n+x-field.offsets()[1]) +
                        10*(y-field.offsets()[2]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from same tile id 3
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[1]) + 
                        10*(y-field.offsets()[2]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top right     - expect from neighbor tile +x id 2
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*2 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(y-field.offsets()[2]-n);
                    EXPECT_EQ(value, expected);
                }
            }
        }
}

template<typename Field>
void check_even_2(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[3]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect from neighbor tile -x id 3
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*3 + 1000*c + z +
                        100*(n-(y-field.offsets()[2]+n)-1) +
                        10*(n+x-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from same tile id 0
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*0 + 1000*c + z +
                        100*(x-field.offsets()[1]) + 
                        10*(n+y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect from same tile id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(n+y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from neighbor tile -x id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*2 + 1000*c + z +
                        100*(n-(y-field.offsets()[2])-1) +
                        10*(n+x-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check right         - expect from same tile id 3
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect empty
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected = -1;
                    EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from neighbor tile +y id 2
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*2 + 1000*c + z +
                        100*(y-field.offsets()[2]-n) +
                        10*(n-(x-field.offsets()[1])-1);
                    EXPECT_EQ(value, expected);
                }
                // check top right     - expect from neighbor tile +y id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*0 + 1000*c + z +
                        100*(y-field.offsets()[2]-n) +
                        10*(n-(x-field.offsets()[1]-n)-1);
                    EXPECT_EQ(value, expected);
                }
            }
        }
}

template<typename Field>
void check_even_3(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[3]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect same tile id 0
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*0 + 1000*c + z +
                        100*(n+x-field.offsets()[1]) + 
                        10*(n+y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from same tile id 1
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[1]) + 
                        10*(n+y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect neighbor tile +x id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*0 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(n+y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from same tile id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                        100*(n+x-field.offsets()[1]) +
                        10*(y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
                // check right         - expect from neighbor tile +x id 2
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*2 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from neighbor tile +y id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*2 + 1000*c + z +
                        100*(y-field.offsets()[2]-n) +
                        10*(n-(n+x-field.offsets()[1])-1);
                    EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from neighbor tile +y id 0
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*0 + 1000*c + z +
                        100*(y-field.offsets()[2]-n) +
                        10*(n-(x-field.offsets()[1])-1);
                    EXPECT_EQ(value, expected);
                }
                // check top right     - expect empty
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected = -1;
                    EXPECT_EQ(value, expected);
                }
            }
        }
}

// odd checks
// -----------

template<typename Field>
void check_odd_0(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[3]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect empty
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected = -1;
                    EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from neighbor tile -y id 3
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*2 + 1000*c + z +
                    //    100*(x-field.offsets()[1]) + 
                    //    10*(n+y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect from neighbor tile -y id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*3 + 1000*c + z +
                    //    100*(x-field.offsets()[1]-n) + 
                    //    10*(n+y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from neighbor tile -x id 1
                for (int x=x_min; x<x_dom_min; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*3 + 1000*c + z +
                    //    100*(n-(y-field.offsets()[2])-1) +
                    //    10*(n+x-field.offsets()[1]);
                    //EXPECT_EQ(value, expected);
                }
                // check right         - expect from same tile id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(y-field.offsets()[2]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from neighbor tile -x id 3
                for (int x=x_min; x<x_dom_min; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*2 + 1000*c + z +
                    //    100*(n-(y-field.offsets()[2]-n)-1) +
                    //    10*(n+x-field.offsets()[1]);
                    //EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from same tile id 2
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                        100*(x-field.offsets()[1]) + 
                        10*(y-field.offsets()[2]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top right     - expect from same tile id 3
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[1]-n) + 
                        10*(y-field.offsets()[2]-n);
                    EXPECT_EQ(value, expected);
                }
            }
        }
}

template<typename Field>
void check_odd_1(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[3]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect from neighbor tile -y id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*2 + 1000*c + z +
                    //    100*(n+x-field.offsets()[1]) + 
                    //    10*(n+y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from neighbor tile -y id 3
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*3 + 1000*c + z +
                    //    100*(x-field.offsets()[1]) + 
                    //    10*(n+y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect empty
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected = -1;
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from same tile id 0
                for (int x=x_min; x<x_dom_min; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(field.domain_id().tile+1) + 10000*0 + 1000*c + z +
                    //    100*(n+x-field.offsets()[1]) +
                    //    10*(y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
                // check right         - expect from neighbor tile +x id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*0 + 1000*c + z +
                    //    100*(x-field.offsets()[1]-n) + 
                    //    10*(y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from same tile id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                    //    100*(n+x-field.offsets()[1]) +
                    //    10*(y-field.offsets()[2]-n);
                    //EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from same tile id 3
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                    //    100*(x-field.offsets()[1]) + 
                    //    10*(y-field.offsets()[2]-n);
                    //EXPECT_EQ(value, expected);
                }
                // check top right     - expect from neighbor tile +x id 2
                for (int x=x_dom_max; x<x_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*2 + 1000*c + z +
                    //    100*(x-field.offsets()[1]-n) + 
                    //    10*(y-field.offsets()[2]-n);
                    //EXPECT_EQ(value, expected);
                }
            }
        }
}

template<typename Field>
void check_odd_2(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[3]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect from neighbor tile -x id 3
                for (int x=x_min; x<x_dom_min; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*3 + 1000*c + z +
                    //    100*(n-(y-field.offsets()[2]+n)-1) +
                    //    10*(n+x-field.offsets()[1]);
                    //EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from same tile id 0
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(field.domain_id().tile+1) + 10000*0 + 1000*c + z +
                    //    100*(x-field.offsets()[1]) + 
                    //    10*(n+y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect from same tile id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                    //    100*(x-field.offsets()[1]-n) + 
                    //    10*(n+y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from neighbor tile -x id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*2 + 1000*c + z +
                    //    100*(n-(y-field.offsets()[2])-1) +
                    //    10*(n+x-field.offsets()[1]);
                    //EXPECT_EQ(value, expected);
                }
                // check right         - expect from same tile id 3
                for (int x=x_dom_max; x<x_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                    //    100*(x-field.offsets()[1]-n) + 
                    //    10*(y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect empty
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected = -1;
                    EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from neighbor tile +y id 2
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*2 + 1000*c + z +
                    //    100*(y-field.offsets()[2]-n) +
                    //    10*(n-(x-field.offsets()[1])-1);
                    //EXPECT_EQ(value, expected);
                }
                // check top right     - expect from neighbor tile +y id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*0 + 1000*c + z +
                    //    100*(y-field.offsets()[2]-n) +
                    //    10*(n-(x-field.offsets()[1]-n)-1);
                    //EXPECT_EQ(value, expected);
                }
            }
        }
}

template<typename Field>
void check_odd_3(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[3]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect same tile id 0
                for (int x=x_min; x<x_dom_min; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(field.domain_id().tile+1) + 10000*0 + 1000*c + z +
                    //    100*(n+x-field.offsets()[1]) + 
                    //    10*(n+y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from same tile id 1
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                    //    100*(x-field.offsets()[1]) + 
                    //    10*(n+y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect neighbor tile +x id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*0 + 1000*c + z +
                    //    100*(x-field.offsets()[1]-n) + 
                    //    10*(n+y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from same tile id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                    //    100*(n+x-field.offsets()[1]) +
                    //    10*(y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
                // check right         - expect from neighbor tile +x id 2
                for (int x=x_dom_max; x<x_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*2 + 1000*c + z +
                    //    100*(x-field.offsets()[1]-n) + 
                    //    10*(y-field.offsets()[2]);
                    //EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from neighbor tile +y id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*2 + 1000*c + z +
                    //    100*(y-field.offsets()[2]-n) +
                    //    10*(n-(n+x-field.offsets()[1])-1);
                    //EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from neighbor tile +y id 0
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    //GHEX_CS_CHECK_VALUE
                    //const value_type expected =
                    //    100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*0 + 1000*c + z +
                    //    100*(y-field.offsets()[2]-n) +
                    //    10*(n-(x-field.offsets()[1])-1);
                    //EXPECT_EQ(value, expected);
                }
                // check top right     - expect empty
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected = -1;
                    EXPECT_EQ(value, expected);
                }
            }
        }
}


template<typename Field>
void check_field(const Field& field, int halo, int n) {
    if (field.domain_id().tile % 2 == 0) {
        switch (field.domain_id().id) {
            case 0:
                check_even_0(field, halo, n);
                break;
            case 1:
                check_even_1(field, halo, n);
                break;
            case 2:
                check_even_2(field, halo, n);
                break;
            case 3:
                check_even_3(field, halo, n);
                break;
        }
    }
    else {
        switch (field.domain_id().id) {
            case 0:
                check_odd_0(field, halo, n);
                break;
            case 1:
                check_odd_1(field, halo, n);
                break;
            case 2:
                check_odd_2(field, halo, n);
                break;
            case 3:
                check_odd_3(field, halo, n);
                break;
        }
    }
}


TEST(halo_generator, domain)
{
    using namespace gridtools::ghex::cubed_sphere;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    
    //domain_descriptor domain(20, 1, 0, std::array<int,3>{10,10,10}, std::array<int,3>{19,19,19});
    //domain_descriptor domain(20, 1, 0, std::array<int,3>{0,0,0}, std::array<int,3>{9,9,9});
    domain_descriptor domain(10, context.rank(), 0, std::array<int,3>{0,0,0}, std::array<int,3>{9,9,5});
    //domain_descriptor domain(10, context.rank()==0?1:3, 0, std::array<int,3>{0,0,0}, std::array<int,3>{9,9,9});
    halo_generator halo_gen(2);
    
    domain_descriptor domain0 (10, context.rank(), 0, std::array<int,3>{0,0,0}, std::array<int,3>{4,4,5});
    domain_descriptor domain1 (10, context.rank(), 1, std::array<int,3>{5,0,0}, std::array<int,3>{9,4,5});
    domain_descriptor domain2 (10, context.rank(), 2, std::array<int,3>{0,5,0}, std::array<int,3>{4,9,5});
    domain_descriptor domain3 (10, context.rank(), 3, std::array<int,3>{5,5,0}, std::array<int,3>{9,9,5});


    const int halo=3;
    std::vector<float> data_dom_0((2*halo+5)*(2*halo+5)*6*8,-1); // fields
    std::vector<float> data_dom_1((2*halo+5)*(2*halo+5)*6*8,-1); // fields
    std::vector<float> data_dom_2((2*halo+5)*(2*halo+5)*6*8,-1); // fields
    std::vector<float> data_dom_3((2*halo+5)*(2*halo+5)*6*8,-1); // fields
    //std::vector<float> large_buffer(data_dom_0.size(), -1);

    field_descriptor<float,gridtools::ghex::cpu> field_dom_0(
        domain0,
        data_dom_0.data(),
        std::array<int,4>{0,halo,halo,0},
        std::array<int,4>{8,2*halo+5,2*halo+5,6});
    field_descriptor<float,gridtools::ghex::cpu> field_dom_1(
        domain1,
        data_dom_1.data(),
        std::array<int,4>{0,halo,halo,0},
        std::array<int,4>{8,2*halo+5,2*halo+5,6});
    field_descriptor<float,gridtools::ghex::cpu> field_dom_2(
        domain2,
        data_dom_2.data(),
        std::array<int,4>{0,halo,halo,0},
        std::array<int,4>{8,2*halo+5,2*halo+5,6});
    field_descriptor<float,gridtools::ghex::cpu> field_dom_3(
        domain3,
        data_dom_3.data(),
        std::array<int,4>{0,halo,halo,0},
        std::array<int,4>{8,2*halo+5,2*halo+5,6});

    for (int comp=0; comp<8; ++comp)
        for (int z=0; z<6; ++z)
            for (int y=0; y<5; ++y)
                for (int x=0; x<5; ++x)
                {
                    const auto idx =
                    (x+halo) +
                    (y+halo)*(2*halo+5) +
                    z*(2*halo+5)*(2*halo+5) +
                    comp*(2*halo+5)*(2*halo+5)*6;
                    data_dom_0[idx] =
                        100000*(domain0.domain_id().tile+1) +
                         10000*domain0.domain_id().id +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                    data_dom_1[idx] =
                        100000*(domain1.domain_id().tile+1) +
                         10000*domain1.domain_id().id +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                    data_dom_2[idx] =
                        100000*(domain2.domain_id().tile+1) +
                         10000*domain2.domain_id().id +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                    data_dom_3[idx] =
                        100000*(domain3.domain_id().tile+1) +
                         10000*domain3.domain_id().id +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                }
    
    //std::vector<domain_descriptor> local_domains{ domain };
    std::vector<domain_descriptor> local_domains{ domain0, domain1, domain2, domain3 };
    auto pattern1 = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains);

    using pattern_type = decltype(pattern1);

    //MPI_Barrier(context.mpi_comm());
    //for (int r=0; r<context.size(); ++r) {
    //    if (r==context.rank()) {
    //        std::cout << "rank " << context.rank() << std::endl;
    //        for (int pp=0; pp<4; ++pp) {
    //            std::cout << "--tile = " << context.rank() << ", id = " << pp << std::endl;
    //        for (const auto& kvp : pattern1[pp].send_halos()) {
    //            const auto& ext_dom_id = kvp.first;
    //            const auto& idx_cont = kvp.second;
    //            std::cout << "  sending to tile " << ext_dom_id.id.tile << ", id " << ext_dom_id.id.id << " on rank " 
    //            << ext_dom_id.mpi_rank << " with tag " << ext_dom_id.tag << std::endl;
    //            for (const auto& isp : idx_cont) {
    //                std::cout 
    //                    << "    iteration space \n"
    //                    << "      global: " << "                  " << isp.global().last() << "\n"
    //                    << "              " << isp.global().first() << "\n"
    //                    << "      local:  " << "                  " << isp.local().last() << "\n"
    //                    << "              " << isp.local().first() << std::endl;
    //            }
    //            //if (r==1 && pp==3) {
    //            //    field_dom_3.pack(large_buffer.data(), idx_cont, nullptr);
    //            //}
    //        }
    //        }
    //    }
    //    MPI_Barrier(context.mpi_comm());
    //}
    //MPI_Barrier(context.mpi_comm());
    //for (int r=0; r<context.size(); ++r) {
    //    if (r==context.rank()) {
    //        std::cout << "rank " << context.rank() << std::endl;
    //        for (int pp=0; pp<4; ++pp) {
    //            std::cout << "--tile = " << context.rank() << ", id = " << pp << std::endl;
    //        for (const auto& kvp : pattern1[pp].recv_halos()) {
    //            const auto& ext_dom_id = kvp.first;
    //            const auto& idx_cont = kvp.second;
    //            std::cout << "  receiving from tile " << ext_dom_id.id.tile << ", id " << ext_dom_id.id.id << " on rank " 
    //            << ext_dom_id.mpi_rank << " with tag " << ext_dom_id.tag << std::endl;
    //            for (const auto& isp : idx_cont) {
    //                std::cout 
    //                    << "    iteration space \n"
    //                    << "      global: " << "                  " << isp.global().last() << "\n"
    //                    << "              " << isp.global().first() << "\n"
    //                    << "      local:  " << "                  " << isp.local().last() << "\n"
    //                    << "              " << isp.local().first() << std::endl;
    //            }
    //            //if (r==1 && pp==3) {
    //            //    field_dom_3.unpack(large_buffer.data(), idx_cont, nullptr);
    //            //}
    //        }
    //        }
    //    }
    //    MPI_Barrier(context.mpi_comm());
    //}

    auto co = gridtools::ghex::make_communication_object<pattern_type>(
        context.get_communicator(context.get_token()));
    
    co.exchange(
        pattern1(field_dom_0),
        pattern1(field_dom_1),
        pattern1(field_dom_2),
        pattern1(field_dom_3)).wait();

    check_field(field_dom_0, 2, 5);
    check_field(field_dom_1, 2, 5);
    //if (context.rank() == 0)
    check_field(field_dom_2, 2, 5);
    check_field(field_dom_3, 2, 5);
    
}
