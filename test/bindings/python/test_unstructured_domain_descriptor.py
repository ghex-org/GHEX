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
import sys
import os

# Debug: log to file to ensure output appears
debug_log_path = os.path.join(os.getcwd(), "ucx_test_debug.log")
debug_log = open(debug_log_path, "a")
print(
    f"[MODULE] test_unstructured_domain_descriptor.py loaded, PID={os.getpid()}, CWD={os.getcwd()}",
    file=debug_log,
    flush=True,
)
debug_log.close()

try:
    import cupy as cp

    # Mock to implement CUDA's Stream protocol: https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
    class CUDAStreamProtocolMock:
        def __init__(self, *args, **kwargs):
            self.cupy_stream = cp.cuda.Stream(*args, **kwargs)

        def __cuda_stream__(self):
            return 0, self.cupy_stream.ptr

    STREAM_TYPES_TO_TEST = [None, cp.cuda.Stream, CUDAStreamProtocolMock]

except ImportError:
    cp = None
    STREAM_TYPES_TO_TEST = [None]  # Must be at least one element.

import ghex
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


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("on_gpu", [True, False])
@pytest.mark.mpi
def test_domain_descriptor(on_gpu, capsys, mpi_cart_comm, cart_context, dtype):
    # Does not uses streams.

    # Debug logging
    with open("/tmp/ucx_test_debug.log", "a") as f:
        print(
            f"[TEST] test_domain_descriptor called: dtype={dtype}, on_gpu={on_gpu}",
            file=f,
            flush=True,
        )

    if on_gpu and cp is None:
        pytest.skip(reason="`CuPy` is not installed.")

    ctx = cart_context
    with open("/tmp/ucx_test_debug.log", "a") as f:
        print(f"[RANK {ctx.rank()}] Starting test_domain_descriptor", file=f, flush=True)
    assert ctx.size() == 4

    domain_desc = DomainDescriptor(
        ctx.rank(), domains[ctx.rank()]["all"], domains[ctx.rank()]["outer_lids"]
    )

    assert domain_desc.domain_id() == ctx.rank()
    assert domain_desc.size() == len(domains[ctx.rank()]["all"])
    assert domain_desc.inner_size() == len(domains[ctx.rank()]["inner"])

    def make_field(order):
        # Creation is always on host.
        data = np.zeros([len(domains[ctx.rank()]["all"]), LEVELS], dtype=dtype, order=order)
        inner_set = set(domains[ctx.rank()]["inner"])
        all_list = domains[ctx.rank()]["all"]
        for x in range(len(all_list)):
            gid = all_list[x]
            for l in range(LEVELS):
                if gid in inner_set:
                    data[x, l] = ctx.rank() * 1000 + 10 * gid + l
                else:
                    data[x, l] = -1

        if on_gpu:
            data = cp.array(data, order=order)

        field = make_field_descriptor(domain_desc, data)
        return data, field

    def check_field(data, order):
        if on_gpu:
            # NOTE: Without the explicit order it fails sometimes.
            data = cp.asnumpy(data, order=order)
        inner_set = set(domains[ctx.rank()]["inner"])
        all_list = domains[ctx.rank()]["all"]
        for x in range(len(all_list)):
            gid = all_list[x]
            for l in range(LEVELS):
                if gid in inner_set:
                    assert data[x, l] == ctx.rank() * 1000 + 10 * gid + l
                else:
                    assert (data[x, l] - 1000 * int((data[x, l]) / 1000)) == 10 * gid + l

        # TODO: Find out if there is a side effect that makes it important to keep them.
        # field = make_field_descriptor(domain_desc, data)
        # return data, field

    halo_gen = HaloGenerator.from_gids(domains[ctx.rank()]["outer"])
    print(f"[RANK {ctx.rank()}] Creating pattern", flush=True)
    pattern = make_pattern(ctx, halo_gen, [domain_desc])
    print(f"[RANK {ctx.rank()}] Creating communication_object", flush=True)
    co = make_communication_object(ctx)

    d1, f1 = make_field("C")
    d2, f2 = make_field("F")

    print(f"[RANK {ctx.rank()}] Calling exchange", flush=True)
    handle = co.exchange([pattern(f1), pattern(f2)])
    print(f"[RANK {ctx.rank()}] Waiting for handle", flush=True)
    handle.wait()
    print(f"[RANK {ctx.rank()}] Handle wait complete", flush=True)

    check_field(d1, "C")
    check_field(d2, "F")
    print(f"[RANK {ctx.rank()}] Test complete", flush=True)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("on_gpu", [True, False])
@pytest.mark.parametrize("stream_type", STREAM_TYPES_TO_TEST)
@pytest.mark.mpi
def test_domain_descriptor_async(on_gpu, stream_type, capsys, mpi_cart_comm, cart_context, dtype):

    if on_gpu:
        if cp is None:
            pytest.skip(reason="`CuPy` is not installed.")
        if not cp.is_available():
            pytest.skip(reason="`CuPy` is installed but no GPU could be found.")
    if not ghex.__config__["gpu"]:
        pytest.skip(
            reason="Skipping `schedule_exchange()` tests because `GHEX` was not compiled with GPU support"
        )

    ctx = cart_context
    print(
        f"[RANK {ctx.rank()}] test_domain_descriptor_async: dtype={dtype}, on_gpu={on_gpu}, stream_type={stream_type}",
        flush=True,
    )
    assert ctx.size() == 4

    domain_desc = DomainDescriptor(
        ctx.rank(), domains[ctx.rank()]["all"], domains[ctx.rank()]["outer_lids"]
    )

    assert domain_desc.domain_id() == ctx.rank()
    assert domain_desc.size() == len(domains[ctx.rank()]["all"])
    assert domain_desc.inner_size() == len(domains[ctx.rank()]["inner"])

    def make_field(order):
        data = np.zeros([len(domains[ctx.rank()]["all"]), LEVELS], dtype=dtype, order=order)
        inner_set = set(domains[ctx.rank()]["inner"])
        all_list = domains[ctx.rank()]["all"]
        for x in range(len(all_list)):
            gid = all_list[x]
            for l in range(LEVELS):
                if gid in inner_set:
                    data[x, l] = ctx.rank() * 1000 + 10 * gid + l
                else:
                    data[x, l] = -1
        if on_gpu:
            data = cp.array(data, order=order)

        field = make_field_descriptor(domain_desc, data)
        return data, field

    def check_field(data, order, stream):
        inner_set = set(domains[ctx.rank()]["inner"])
        all_list = domains[ctx.rank()]["all"]
        if on_gpu:
            # NOTE: Without the explicit order it fails sometimes.
            data = cp.asnumpy(data, order=order, stream=stream, blocking=True)

        for x in range(len(all_list)):
            gid = all_list[x]
            for l in range(LEVELS):
                if gid in inner_set:
                    assert data[x, l] == ctx.rank() * 1000 + 10 * gid + l
                else:
                    assert (data[x, l] - 1000 * int((data[x, l]) / 1000)) == 10 * gid + l

    halo_gen = HaloGenerator.from_gids(domains[ctx.rank()]["outer"])
    pattern = make_pattern(ctx, halo_gen, [domain_desc])
    co = make_communication_object(ctx)

    d1, f1 = make_field("C")
    d2, f2 = make_field("F")

    stream = None if stream_type is None else stream_type(non_blocking=True)
    print(f"[RANK {ctx.rank()}] Calling schedule_exchange", flush=True)
    handle = co.schedule_exchange(stream, [pattern(f1), pattern(f2)])
    print(f"[RANK {ctx.rank()}] schedule_exchange returned", flush=True)
    assert not co.has_scheduled_exchange()

    print(f"[RANK {ctx.rank()}] Calling schedule_wait", flush=True)
    handle.schedule_wait(stream)
    print(f"[RANK {ctx.rank()}] schedule_wait returned", flush=True)
    assert co.has_scheduled_exchange()

    check_field(d1, "C", stream)
    check_field(d2, "F", stream)
    assert co.has_scheduled_exchange()

    print(f"[RANK {ctx.rank()}] Calling handle.wait()", flush=True)
    handle.wait()
    print(f"[RANK {ctx.rank()}] handle.wait() returned", flush=True)
    assert not co.has_scheduled_exchange()
    print(f"[RANK {ctx.rank()}] Async test complete", flush=True)
