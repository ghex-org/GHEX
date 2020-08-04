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
#include <ghex/structured/cubed_sphere/halo_generator.hpp>
#include <ghex/structured/cubed_sphere/field_descriptor.hpp>

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
// neigborhood of a tile/face is depicted below (tile in the center, neighbors around in -y,-x,+x,+y directions)
//              
// -------------------------------------------------------+-------------------------------------------------------
//                       even tiles                       :                      odd tiles                        
// -------------------------------------------------------+-------------------------------------------------------
//                   +--------+--------+                  :                  +--------+--------+                  
//                   |        |        |                  :                  |        |        |                  
//                   |        |        |                  :                  |        |        |                  
//                   |       3|1       |                  :                  |       2|3       |                  
//                   +--------+--------+                  :                  +--------+--------+                  
//                   |       2|0   x ∧ |                  :                  | ∧ y   0|1       |                  
//                   |        |      | |                  :                  | |      |        |                  
//                   |        |  y<--+ |                  :                  | +-->x  |        |                  
// +--------+--------+--------+--------+--------+--------+:+--------+--------+--------+--------+--------+--------+
// | +-->y  |        |        |        |        |        |:|        |        |        |        | +-->y  |        |
// | |      |        |        |        |        |        |:|        |        |        |        | |      |        |
// | ∨ x   0|2       |       2|3       |       2|3       |:|       2|3       |       2|3       | ∨ x   0|3       |
// +--------+--------+--------+--------+--------+--------+:+--------+--------+--------+--------+--------+--------+
// |       1|3       | ∧ y   0|1       | ∧ y   0|1       |:| ∧ y   0|1       | ∧ y   0|1       |       1|2       |
// |        |        | |      |        | |      |        |:| |      |        | |      |        |        |        |
// |        |        | +-->x  |        | +-->x  |        |:| +-->x  |        | +-->x  |        |        |        |
// +--------+--------+--------+--------+--------+--------+:+--------+--------+--------+--------+--------+--------+
//                   |        |        |                  :                  |        |        |                  
//                   |        |        |                  :                  |        |        |                  
//                   |       2|3       |                  :                  |       3|1       |                  
//                   +--------+--------+                  :                  +--------+--------+                  
//                   | ∧ y   0|1       |                  :                  |       2|0   x ∧ |                  
//                   | |      |        |                  :                  |        |      | |                  
//                   | +-->x  |        |                  :                  |        |  y<--+ |                  
//                   +--------+--------+                  :                  +--------+--------+                  
// -------------------------------------------------------+-------------------------------------------------------

// helper macro for checks
#define GHEX_CS_CHECK_HEADER                      \
    const auto x_dom_min = field.offsets()[0];    \
    const auto x_min     = x_dom_min-halo;        \
    const auto y_dom_min = field.offsets()[1];    \
    const auto y_min     = y_dom_min-halo;        \
    const auto x_dom_max = x_dom_min + n;         \
    const auto x_max     = x_dom_max+halo;        \
    const auto y_dom_max = y_dom_min + n;         \
    const auto y_max     = y_dom_max+halo;        \
    const auto strides   = field.byte_strides();  \
    using value_type = typename Field::value_type;

// helper macro for checks
#define GHEX_CS_CHECK_VALUE                                                                \
    const auto memory_location = strides[3]*c + strides[0]*x + strides[1]*y+ strides[2]*z; \
    const value_type value = *reinterpret_cast<const value_type*>(                         \
        reinterpret_cast<const char*>(field.data())+memory_location);

template<typename Id>
int id_to_int(const Id& id) {
    if (id[0]==0 && id[1]==0) return 0;
    else if (id[1]==0) return 1;
    else if (id[0]==0) return 2;
    else return 3;
}

// even checks
// -----------

// check received data for even tile and subdomain with id 0
template<typename Field>
void check_even_0(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::structured::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[2]; ++z) {
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
                        100*(x-field.offsets()[0]) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect from neighbor tile -y id 3
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from neighbor tile -x id 3
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*3 + 1000*c + z +
                        100*(n-(y-field.offsets()[1])-1) +
                        10*(n+x-field.offsets()[0]);
                    const value_type v_factor = (field.is_vector_field() && c==1) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
                // check right         - expect from same tile id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from neighbor tile -x id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*2 + 1000*c + z +
                        100*(n-(y-field.offsets()[1]-n)-1) +
                        10*(n+x-field.offsets()[0]);
                    const value_type v_factor = (field.is_vector_field() && c==1) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
                // check top middle    - expect from same tile id 2
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                        100*(x-field.offsets()[0]) + 
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top right     - expect from same tile id 3
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
            }
        }
}

// check received data for even tile and subdomain with id 1
template<typename Field>
void check_even_1(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::structured::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[2]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect from neighbor tile -y id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*2 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from neighbor tile -y id 3
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[0]) + 
                        10*(n+y-field.offsets()[1]);
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
                        100*(n+x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check right         - expect from neighbor tile +x id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*0 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from same tile id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from same tile id 3
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[0]) + 
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top right     - expect from neighbor tile +x id 2
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*2 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
            }
        }
}

// check received data for even tile and subdomain with id 2
template<typename Field>
void check_even_2(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::structured::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[2]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect from neighbor tile -x id 3
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*3 + 1000*c + z +
                        100*(n-(y-field.offsets()[1]+n)-1) +
                        10*(n+x-field.offsets()[0]);
                    const value_type v_factor = (field.is_vector_field() && c==1) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
                // check bottom middle - expect from same tile id 0
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*0 + 1000*c + z +
                        100*(x-field.offsets()[0]) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect from same tile id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from neighbor tile -x id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*2 + 1000*c + z +
                        100*(n-(y-field.offsets()[1])-1) +
                        10*(n+x-field.offsets()[0]);
                    const value_type v_factor = (field.is_vector_field() && c==1) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
                // check right         - expect from same tile id 3
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(y-field.offsets()[1]);
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
                        100*(y-field.offsets()[1]-n) +
                        10*(n-(x-field.offsets()[0])-1);
                    const value_type v_factor = (field.is_vector_field() && c==0) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
                // check top right     - expect from neighbor tile +y id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*0 + 1000*c + z +
                        100*(y-field.offsets()[1]-n) +
                        10*(n-(x-field.offsets()[0]-n)-1);
                    const value_type v_factor = (field.is_vector_field() && c==0) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
            }
        }
}

// check received data for even tile and subdomain with id 3
template<typename Field>
void check_even_3(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::structured::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[2]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect same tile id 0
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*0 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from same tile id 1
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[0]) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect neighbor tile +x id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*0 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from same tile id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check right         - expect from neighbor tile +x id 2
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*2 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from neighbor tile +y id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*2 + 1000*c + z +
                        100*(y-field.offsets()[1]-n) +
                        10*(n-(n+x-field.offsets()[0])-1);
                    const value_type v_factor = (field.is_vector_field() && c==0) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
                // check top middle    - expect from neighbor tile +y id 0
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*0 + 1000*c + z +
                        100*(y-field.offsets()[1]-n) +
                        10*(n-(x-field.offsets()[0])-1);
                    const value_type v_factor = (field.is_vector_field() && c==0) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
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

// check received data for odd tile and subdomain with id 0
template<typename Field>
void check_odd_0(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::structured::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[2]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect empty
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected = -1;
                    EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from neighbor tile -y id 3
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*3 + 1000*c + z +
                        100*(n+y-field.offsets()[1]) + 
                        10*(n-(x-field.offsets()[0])-1);
                    const value_type v_factor = (field.is_vector_field() && c==0) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
                // check bottom right  - expect from neighbor tile -y id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*1 + 1000*c + z +
                        100*(n+y-field.offsets()[1]) + 
                        10*(n-(x-field.offsets()[0]-n)-1);
                    const value_type v_factor = (field.is_vector_field() && c==0) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from neighbor tile -x id 1
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*1 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check right         - expect from same tile id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from neighbor tile -x id 3
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*3 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from same tile id 2
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                        100*(x-field.offsets()[0]) + 
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top right     - expect from same tile id 3
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
            }
        }
}

// check received data for odd tile and subdomain with id 1
template<typename Field>
void check_odd_1(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::structured::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[2]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect from neighbor tile -y id 3
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*3 + 1000*c + z +
                        100*(n+y-field.offsets()[1]) + 
                        10*(n-(x-field.offsets()[0]+n)-1);
                    const value_type v_factor = (field.is_vector_field() && c==0) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
                // check bottom middle - expect from neighbor tile -y id 1
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][2]+1) + 10000*1 + 1000*c + z +
                        100*(n+y-field.offsets()[1]) + 
                        10*(n-(x-field.offsets()[0])-1);
                    const value_type v_factor = (field.is_vector_field() && c==0) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
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
                        100*(n+x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check right         - expect from neighbor tile +x id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*1 + 1000*c + z +
                        100*(n-(y-field.offsets()[1])-1) + 
                        10*(x-field.offsets()[0]-n);
                    const value_type v_factor = (field.is_vector_field() && c==1) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from same tile id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from same tile id 3
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[0]) + 
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top right     - expect from neighbor tile +x id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*0 + 1000*c + z +
                        100*(n-(y-field.offsets()[1]-n)-1) + 
                        10*(x-field.offsets()[0]-n);
                    const value_type v_factor = (field.is_vector_field() && c==1) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
            }
        }
}

// check received data for odd tile and subdomain with id 2
template<typename Field>
void check_odd_2(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::structured::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[2]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect from neighbor tile -x id 1
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*1 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) +
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from same tile id 0
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*0 + 1000*c + z +
                        100*(x-field.offsets()[0]) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect from same tile id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from neighbor tile -x id 3
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][0]+1) + 10000*3 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check right         - expect from same tile id 3
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*3 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) + 
                        10*(y-field.offsets()[1]);
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
                // check top middle    - expect from neighbor tile +y id 0
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*0 + 1000*c + z +
                        100*(x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top right     - expect from neighbor tile +y id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[0]-n) +
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
            }
        }
}

// check received data for odd tile and subdomain with id 3
template<typename Field>
void check_odd_3(const Field& field, int halo, int n) {
    GHEX_CS_CHECK_HEADER
    using namespace gridtools::ghex::structured::cubed_sphere;
    for (int c=0; c<field.num_components(); ++c)
        for (int z=0; z<field.extents()[2]; ++z) {
            for (int y=y_min; y<y_dom_min; ++y) {
                // check bottom left   - expect same tile id 0
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*0 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom middle - expect from same tile id 1
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[0]) + 
                        10*(n+y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check bottom right  - expect neighbor tile +x id 1
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*1 + 1000*c + z +
                        100*(n-(y-field.offsets()[1]+n)-1) + 
                        10*(x-field.offsets()[0]-n);
                    const value_type v_factor = (field.is_vector_field() && c==1) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
            }
            for (int y=y_dom_min; y<y_dom_max; ++y) {
                // check left          - expect from same tile id 2
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(field.domain_id().tile+1) + 10000*2 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]);
                    EXPECT_EQ(value, expected);
                }
                // check right         - expect from neighbor tile +x id 0
                for (int x=x_dom_max; x<x_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][1]+1) + 10000*0 + 1000*c + z +
                        100*(n-(y-field.offsets()[1])-1) + 
                        10*(x-field.offsets()[0]-n);
                    const value_type v_factor = (field.is_vector_field() && c==1) ? -1 : 1;
                    EXPECT_EQ(value, v_factor*expected);
                }
            }
            for (int y=y_dom_max; y<y_max; ++y) {
                // check top left      - excpect from neighbor tile +y id 0
                for (int x=x_min; x<x_dom_min; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*0 + 1000*c + z +
                        100*(n+x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]-n);
                    EXPECT_EQ(value, expected);
                }
                // check top middle    - expect from neighbor tile +y id 1
                for (int x=x_dom_min; x<x_dom_max; ++x) {
                    GHEX_CS_CHECK_VALUE
                    const value_type expected =
                        100000*(tile_lu[field.domain_id().tile][3]+1) + 10000*1 + 1000*c + z +
                        100*(x-field.offsets()[0]) +
                        10*(y-field.offsets()[1]-n);
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

// check received data
template<typename Field>
void check_field(const Field& field, int halo, int n) {
    const auto id = id_to_int(field.domain_id().id);
    if (field.domain_id().tile % 2 == 0) {
        switch (id) {
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
        switch (id) {
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

TEST(cubed_sphere, domain)
{
    using namespace gridtools::ghex::structured::cubed_sphere;

    // create context
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    
    // halo generator with 2 halo lines in x and y dimensions (on both sides)
    halo_generator halo_gen(2);

    // cube with size 10 and 6 levels
    cube c{10,6};
    
    // define 4 local domains
    domain_descriptor domain0 (c, context.rank(), 0, 4, 0, 4);
    domain_descriptor domain1 (c, context.rank(), 5, 9, 0, 4);
    domain_descriptor domain2 (c, context.rank(), 0, 4, 5, 9);
    domain_descriptor domain3 (c, context.rank(), 5, 9, 5, 9);
    std::vector<domain_descriptor> local_domains{ domain0, domain1, domain2, domain3 };

    // allocate large enough memory for fields, sufficient for 3 halo lines
    // use 8 components per field and 6 z-levels
    const int halo=3;
    std::vector<float> data_dom_0((2*halo+5)*(2*halo+5)*6*8,-1); // fields
    std::vector<float> data_dom_1((2*halo+5)*(2*halo+5)*6*8,-1); // fields
    std::vector<float> data_dom_2((2*halo+5)*(2*halo+5)*6*8,-1); // fields
    std::vector<float> data_dom_3((2*halo+5)*(2*halo+5)*6*8,-1); // fields

    // initialize physical domain (leave halos as they are)
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
                         10000*id_to_int(domain0.domain_id().id) +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                    data_dom_1[idx] =
                        100000*(domain1.domain_id().tile+1) +
                         10000*id_to_int(domain1.domain_id().id) +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                    data_dom_2[idx] =
                        100000*(domain2.domain_id().tile+1) +
                         10000*id_to_int(domain2.domain_id().id) +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                    data_dom_3[idx] =
                        100000*(domain3.domain_id().tile+1) +
                         10000*id_to_int(domain3.domain_id().id) +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                }

#ifdef __CUDACC__
    using arch_t = gridtools::ghex::gpu;
    float* data_ptr_0 = nullptr;
    float* data_ptr_1 = nullptr;
    float* data_ptr_2 = nullptr;
    float* data_ptr_3 = nullptr;
    cudaMalloc((void**)&data_ptr_0, data_dom_0.size()*sizeof(float));
    cudaMalloc((void**)&data_ptr_1, data_dom_1.size()*sizeof(float));
    cudaMalloc((void**)&data_ptr_2, data_dom_2.size()*sizeof(float));
    cudaMalloc((void**)&data_ptr_3, data_dom_3.size()*sizeof(float));
    cudaMemcpy(data_ptr_0, data_dom_0.data(), data_dom_0.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_ptr_1, data_dom_1.data(), data_dom_1.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_ptr_2, data_dom_2.data(), data_dom_2.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_ptr_3, data_dom_3.data(), data_dom_3.size()*sizeof(float), cudaMemcpyHostToDevice);
#else
    using arch_t = gridtools::ghex::cpu;
    float* data_ptr_0 = data_dom_0.data();
    float* data_ptr_1 = data_dom_1.data();
    float* data_ptr_2 = data_dom_2.data();
    float* data_ptr_3 = data_dom_3.data();
#endif

    // wrap field memory in a field_descriptor
    field_descriptor<float,arch_t> field_dom_0(
        domain0,
        data_ptr_0,
        std::array<int,3>{halo,halo,0},
        std::array<int,3>{2*halo+5,2*halo+5,6},
        8);
    field_descriptor<float,arch_t> field_dom_1(
        domain1,
        data_ptr_1,
        std::array<int,3>{halo,halo,0},
        std::array<int,3>{2*halo+5,2*halo+5,6},
        8);
    field_descriptor<float,arch_t> field_dom_2(
        domain2,
        data_ptr_2,
        std::array<int,3>{halo,halo,0},
        std::array<int,3>{2*halo+5,2*halo+5,6},
        8);
    field_descriptor<float,arch_t> field_dom_3(
        domain3,
        data_ptr_3,
        std::array<int,3>{halo,halo,0},
        std::array<int,3>{2*halo+5,2*halo+5,6},
        8);

    // create a structured pattern
    auto pattern1 = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(
        context, halo_gen, local_domains);

    // make a communication object
    using pattern_type = decltype(pattern1);
    auto co = gridtools::ghex::make_communication_object<pattern_type>(
        context.get_communicator(context.get_token()));
    
    // exchange halo data
    co.exchange(
        pattern1(field_dom_0),
        pattern1(field_dom_1),
        pattern1(field_dom_2),
        pattern1(field_dom_3)).wait();

#ifdef __CUDACC__
    cudaMemcpy(data_dom_0.data(), data_ptr_0, data_dom_0.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_dom_1.data(), data_ptr_1, data_dom_1.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_dom_2.data(), data_ptr_2, data_dom_2.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_dom_3.data(), data_ptr_3, data_dom_3.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_ptr_0);
    cudaFree(data_ptr_1);
    cudaFree(data_ptr_2);
    cudaFree(data_ptr_3);
    field_dom_0.set_data(data_dom_0.data());
    field_dom_1.set_data(data_dom_1.data());
    field_dom_2.set_data(data_dom_2.data());
    field_dom_3.set_data(data_dom_3.data());
#endif

    // check results
    check_field(field_dom_0, 2, 5);
    check_field(field_dom_1, 2, 5);
    check_field(field_dom_2, 2, 5);
    check_field(field_dom_3, 2, 5);
}

TEST(cubed_sphere, domain_vector)
{
    using namespace gridtools::ghex::structured::cubed_sphere;

    // create context
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    
    // halo generator with 2 halo lines in x and y dimensions (on both sides)
    halo_generator halo_gen(2);
    
    // cube with size 10 and 7 levels
    cube c{10,7};
    
    // define 4 local domains
    domain_descriptor domain0 (c, context.rank(), 0, 4, 0, 4);
    domain_descriptor domain1 (c, context.rank(), 5, 9, 0, 4);
    domain_descriptor domain2 (c, context.rank(), 0, 4, 5, 9);
    domain_descriptor domain3 (c, context.rank(), 5, 9, 5, 9);
    std::vector<domain_descriptor> local_domains{ domain0, domain1, domain2, domain3 };

    // allocate large enough memory for fields, sufficient for 3 halo lines
    // use 3 components per field and 7 z-levels
    const int halo=3;
    std::vector<float> data_dom_0((2*halo+5)*(2*halo+5)*3*7,-1); // fields
    std::vector<float> data_dom_1((2*halo+5)*(2*halo+5)*3*7,-1); // fields
    std::vector<float> data_dom_2((2*halo+5)*(2*halo+5)*3*7,-1); // fields
    std::vector<float> data_dom_3((2*halo+5)*(2*halo+5)*3*7,-1); // fields

    // initialize physical domain (leave halos as they are)
    for (int comp=0; comp<3; ++comp)
        for (int z=0; z<7; ++z)
            for (int y=0; y<5; ++y)
                for (int x=0; x<5; ++x)
                {
                    const auto idx =
                    (x+halo) +
                    (y+halo)*(2*halo+5) +
                    z*(2*halo+5)*(2*halo+5) +
                    comp*(2*halo+5)*(2*halo+5)*7;
                    data_dom_0[idx] =
                        100000*(domain0.domain_id().tile+1) +
                         10000*id_to_int(domain0.domain_id().id) +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                    data_dom_1[idx] =
                        100000*(domain1.domain_id().tile+1) +
                         10000*id_to_int(domain1.domain_id().id) +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                    data_dom_2[idx] =
                        100000*(domain2.domain_id().tile+1) +
                         10000*id_to_int(domain2.domain_id().id) +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                    data_dom_3[idx] =
                        100000*(domain3.domain_id().tile+1) +
                         10000*id_to_int(domain3.domain_id().id) +
                          1000*comp +
                           100*x +
                            10*y +
                             1*z;
                }

#ifdef __CUDACC__
    using arch_t = gridtools::ghex::gpu;
    float* data_ptr_0 = nullptr;
    float* data_ptr_1 = nullptr;
    float* data_ptr_2 = nullptr;
    float* data_ptr_3 = nullptr;
    cudaMalloc((void**)&data_ptr_0, data_dom_0.size()*sizeof(float));
    cudaMalloc((void**)&data_ptr_1, data_dom_1.size()*sizeof(float));
    cudaMalloc((void**)&data_ptr_2, data_dom_2.size()*sizeof(float));
    cudaMalloc((void**)&data_ptr_3, data_dom_3.size()*sizeof(float));
    cudaMemcpy(data_ptr_0, data_dom_0.data(), data_dom_0.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_ptr_1, data_dom_1.data(), data_dom_1.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_ptr_2, data_dom_2.data(), data_dom_2.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_ptr_3, data_dom_3.data(), data_dom_3.size()*sizeof(float), cudaMemcpyHostToDevice);
#else
    using arch_t = gridtools::ghex::cpu;
    float* data_ptr_0 = data_dom_0.data();
    float* data_ptr_1 = data_dom_1.data();
    float* data_ptr_2 = data_dom_2.data();
    float* data_ptr_3 = data_dom_3.data();
#endif

    // wrap field memory in a field_descriptor
    field_descriptor<float,arch_t> field_dom_0(
        domain0,
        data_ptr_0,
        std::array<int,3>{halo,halo,0},
        std::array<int,3>{2*halo+5,2*halo+5,7},
        3, true);
    field_descriptor<float,arch_t> field_dom_1(
        domain1,
        data_ptr_1,
        std::array<int,3>{halo,halo,0},
        std::array<int,3>{2*halo+5,2*halo+5,7},
        3, true);
    field_descriptor<float,arch_t> field_dom_2(
        domain2,
        data_ptr_2,
        std::array<int,3>{halo,halo,0},
        std::array<int,3>{2*halo+5,2*halo+5,7},
        3, true);
    field_descriptor<float,arch_t> field_dom_3(
        domain3,
        data_ptr_3,
        std::array<int,3>{halo,halo,0},
        std::array<int,3>{2*halo+5,2*halo+5,7},
        3, true);

    // create a structured pattern
    auto pattern1 = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(
        context, halo_gen, local_domains);

    // make a communication object
    using pattern_type = decltype(pattern1);
    auto co = gridtools::ghex::make_communication_object<pattern_type>(
        context.get_communicator(context.get_token()));
    
    // exchange halo data
    co.exchange(
        pattern1(field_dom_0),
        pattern1(field_dom_1),
        pattern1(field_dom_2),
        pattern1(field_dom_3)).wait();

#ifdef __CUDACC__
    cudaMemcpy(data_dom_0.data(), data_ptr_0, data_dom_0.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_dom_1.data(), data_ptr_1, data_dom_1.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_dom_2.data(), data_ptr_2, data_dom_2.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_dom_3.data(), data_ptr_3, data_dom_3.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_ptr_0);
    cudaFree(data_ptr_1);
    cudaFree(data_ptr_2);
    cudaFree(data_ptr_3);
    field_dom_0.set_data(data_dom_0.data());
    field_dom_1.set_data(data_dom_1.data());
    field_dom_2.set_data(data_dom_2.data());
    field_dom_3.set_data(data_dom_3.data());
#endif

    // check results
    check_field(field_dom_0, 2, 5);
    check_field(field_dom_1, 2, 5);
    check_field(field_dom_2, 2, 5);
    check_field(field_dom_3, 2, 5);
}
