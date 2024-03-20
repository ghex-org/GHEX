#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
import pytest
import numpy as np

# import cupy as cp

from ghex.context import make_context
from ghex.unstructured import make_communication_object
from ghex.unstructured import DomainDescriptor
from ghex.unstructured import HaloGenerator
from ghex.unstructured import make_pattern
from ghex.unstructured import make_field_descriptor

# fmt: off
# Exchange of unstructured domain
# This test uses a rectangular domain and divides it unevenly into
# 4 subdomains (4 ranks). Directly neighboring cells are considered
# halo cells (1 level). Includes following corner cases:
# - exchange with itself (periodic wrapping within same subdomain)
# - repeated global indices in outer halo
# - repeated global indices in outer halo in exchange with itself

# global domain:
#        +--------+--------------------+
#        | 0  1  2| 3  4  5  6  7  8  9|
#        |        +--+                 |
#        |10 11 12 13|14 15 16 17 18 19|
#        +--+        +--+              |
#        |20|21 22 23 24|25 26 27 28 29|
#        |  +--------+  +              |
#        |30 31 32 33|34|35 36 37 38 39|
#        |           +--+              |
#        |40 41 42 43 44|45 46 47 48 49|
#        |              |              |
#        |50 51 52 53 54|55 56 57 58 59|
#        +--------------+--+           |
#        |60 61 62 63 64 65|66 67 68 69|
#        |                 |           |
#        |70 71 72 73 74 75|76 77 78 79|
#        +-----------------+-----------+

domains = {
    # subdomain 0:
    #                      79 70 71 72 73
    #                        +--------+
    #                       9| 0  1  2| 3  4
    #                        +        +--+
    #                      19|10 11 12 13|14 15
    #                        +--+        +--+
    #                      29 20|21 22 23 24|25
    #                           +--------+  +
    #                         30 31 32 33|34|35
    #                                    +--+
    #                                  43 44 45
    0: {
        "all":        [79, 70, 71, 72, 73,
                        9,  0,  1,  2,  3, 4,
                       19, 10, 11, 12, 13, 14, 15,
                       29, 20, 21, 22, 23, 24, 25,
                       30, 31, 32, 33, 34, 35,
                                   43, 44, 45],
        "inner":         [ 0,  1,  2,
                          10, 11, 12, 13,
                              21, 22, 23, 24,
                                          34],
        "outer":      [79, 70, 71, 72, 73,
                        9,              3,  4,
                       19,                 14, 15,
                       29, 20,                 25,
                           30, 31, 32, 33,     35,
                                       43, 44, 45],
        "outer_lids": [ 0,  1,  2,  3,  4,
                        5,              9, 10,
                       11,                 16, 17,
                       18, 19,                 24,
                           25, 26, 27, 28,     30,
                                       31, 32, 33],
    },
    # sbudomain 1:
    #                      72 73 74 75 76 77 78 79 70
    #                        +--------------------+
    #                       2| 3  4  5  6  7  8  9| 0
    #                        +--+                 +
    #                      12 13|14 15 16 17 18 19|10
    #                           +--+              +
    #                         23 24|25 26 27 28 29|20
    #                              +              +
    #                            34|35 36 37 38 39|30
    #                              +              +
    #                            44|45 46 47 48 49|40
    #                              +              +
    #                            54|55 56 57 58 59|50
    #                              +--+           +
    #                            64 65|66 67 68 69|60
    #                                 +           +
    #                               75|76 77 78 79|70
    #                                 +-----------+
    #                                5  6  7  8  9  0
    1: {
        "all": [72, 73, 74, 75, 76, 77, 78, 79, 70,
                 2,  3,  4,  5,  6,  7,  8,  9,  0,
                12, 13, 14, 15, 16, 17, 18, 19, 10,
                    23, 24, 25, 26, 27, 28, 29, 20,
                        34, 35, 36, 37, 38, 39, 30,
                        44, 45, 46, 47, 48, 49, 40,
                        54, 55, 56, 57, 58, 59, 50,
                        64, 65, 66, 67, 68, 69, 60,
                            75, 76, 77, 78, 79, 70,
                             5,  6,  7,  8,  9,  0],
        "inner": [ 3,  4,  5,  6,  7,  8,  9,
                      14, 15, 16, 17, 18, 19,
                          25, 26, 27, 28, 29,
                          35, 36, 37, 38, 39,
                          45, 46, 47, 48, 49,
                          55, 56, 57, 58, 59,
                              66, 67, 68, 69,
                              76, 77, 78, 79],
        "outer": [72, 73, 74, 75, 76, 77, 78, 79, 70,
                   2,                              0,
                  12, 13,                         10,
                      23, 24,                     20,
                          34,                     30,
                          44,                     40,
                          54,                     50,
                          64, 65,                 60,
                              75,                 70,
                               5,  6,  7,  8,  9,  0],
        "outer_lids": [ 0,  1,  2,  3,  4,  5,  6,  7,  8,
                        9,                             17,
                       18, 19,                         26,
                           27, 28,                     34,
                               35,                     41,
                               42,                     48,
                               49,                     55,
                               56, 57,                 62,
                                   63,                 68,
                                   69, 70, 71, 72, 73, 74],
    },
    # subdomain 2:
    #                      19 10 11
    #                        +--+
    #                      29|20|21 22 23 24
    #                        +  +--------+
    #                      39|30 31 32 33|34 35
    #                        +           +--+
    #                      49|40 41 42 43 44|45
    #                        +              +
    #                      59|50 51 52 53 54|55
    #                        +--------------+
    #                      69 60 61 62 63 64 65
    2: {
        "all": [19, 10, 11,
                29, 20, 21, 22, 23, 24,
                39, 30, 31, 32, 33, 34, 35,
                49, 40, 41, 42, 43, 44, 45,
                59, 50, 51, 52, 53, 54, 55,
                69, 60, 61, 62, 63, 64, 65],
        "inner": [20,
                  30, 31, 32, 33,
                  40, 41, 42, 43, 44,
                  50, 51, 52, 53, 54],
        "outer": [19, 10, 11,
                  29,     21, 22, 23, 24,
                  39,                 34, 35,
                  49,                     45,
                  59,                     55,
                  69, 60, 61, 62, 63, 64, 65],
        "outer_lids": [ 0,  1,  2,
                        3,      5,  6,  7,  8,
                        9,                 14, 15,
                       16,                     22,
                       23,                     29,
                       30, 31, 32, 33, 34, 35, 36],
    },
    # subdomain 3:
    #                      59 50 51 52 53 54 55 56
    #                        +-----------------+
    #                      69|60 61 62 63 64 65|66
    #                        +                 +
    #                      79|70 71 72 73 74 75|76
    #                        +-----------------+
    #                       9  0  1  2  3  4  5  6
    3: {
        "all": [59, 50, 51, 52, 53, 54, 55, 56,
                69, 60, 61, 62, 63, 64, 65, 66,
                79, 70, 71, 72, 73, 74, 75, 76,
                 9,  0,  1,  2,  3,  4,  5,  6],
        "inner": [60, 61, 62, 63, 64, 65,
                  70, 71, 72, 73, 74, 75],
        "outer": [59, 50, 51, 52, 53, 54, 55, 56,
                  69,                         66,
                  79,                         76,
                   9,  0,  1,  2,  3,  4,  5,  6],
        "outer_lids": [ 0,  1,  2,  3,  4,  5,  6,  7,
                        8,                         15,
                       16,                         23,
                       24, 25, 26, 27, 28, 29, 30, 31],
    },
}
# fmt: on

LEVELS = 2


@pytest.mark.mpi
def test_domain_descriptor(capsys, mpi_cart_comm):
    ctx = make_context(mpi_cart_comm, True)
    assert ctx.size() == 4

    domain_desc = DomainDescriptor(
        ctx.rank(), domains[ctx.rank()]["all"], domains[ctx.rank()]["outer_lids"]
    )

    assert domain_desc.domain_id() == ctx.rank()
    assert domain_desc.size() == len(domains[ctx.rank()]["all"])
    assert domain_desc.inner_size() == len(domains[ctx.rank()]["inner"])

    halo_gen = HaloGenerator.from_gids(domains[ctx.rank()]["outer"])

    pattern = make_pattern(ctx, halo_gen, [domain_desc])

    co = make_communication_object(ctx)

    def make_field(order):
        data = np.zeros(
            [len(domains[ctx.rank()]["all"]), LEVELS], dtype=np.float64, order=order
        )
        inner_set = set(domains[ctx.rank()]["inner"])
        all_list = domains[ctx.rank()]["all"]
        for x in range(len(all_list)):
            gid = all_list[x]
            for l in range(LEVELS):
                if gid in inner_set:
                    data[x, l] = ctx.rank() * 1000 + 10 * gid + l
                else:
                    data[x, l] = -1

        field = make_field_descriptor(domain_desc, data)
        return data, field

    def check_field(data):
        inner_set = set(domains[ctx.rank()]["inner"])
        all_list = domains[ctx.rank()]["all"]
        for x in range(len(all_list)):
            gid = all_list[x]
            for l in range(LEVELS):
                if gid in inner_set:
                    assert data[x, l] == ctx.rank() * 1000 + 10 * gid + l
                else:
                    assert (
                        data[x, l] - 1000 * int((data[x, l]) / 1000)
                    ) == 10 * gid + l

        field = make_field_descriptor(domain_desc, data)
        return data, field

    d1, f1 = make_field("C")
    d2, f2 = make_field("F")

    # np.set_printoptions(precision=8, suppress=True)
    # with capsys.disabled():
    #    print("")
    #    print(d1)

    res = co.exchange([pattern(f1), pattern(f2)])
    res.wait()

    # with capsys.disabled():
    #    print("")
    #    print("")
    #    print("")
    #    print(d1)

    check_field(d1)
    check_field(d2)
