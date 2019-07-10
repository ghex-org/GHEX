/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */

#ifdef __CUDACC__
#include <gridtools/common/cuda_util.hpp>
#endif

#include <vector>
#include <iostream>

#include <gtest/gtest.h>
#include "gtest_main_gpu.cpp"

#include <boost/mpi/communicator.hpp>

TEST(gpu, allocate)
{
    bool passed = true;

    boost::mpi::communicator world;

#ifdef __CUDACC__

    MPI_Comm raw_local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world.rank(), MPI_INFO_NULL, &raw_local_comm);
    boost::mpi::communicator local_comm(raw_local_comm, boost::mpi::comm_take_ownership);

    if (local_comm.rank()==0)
    {
        std::cout << "I am rank " << world.rank() << " and I own GPU " << world.rank()/local_comm.size() << std::endl;
        GT_CUDA_CHECK(cudaSetDevice(0));
        std::cout << "gpu is invoked!" << std::endl;

        std::vector<int> data = {0,1,2,3,4,5,6};
        int* gpu_ptr;
        GT_CUDA_CHECK(cudaMalloc(&gpu_ptr, data.size()*sizeof(int)));
        std::vector<int> data_out(data.size());
        
        GT_CUDA_CHECK(cudaMemcpy(gpu_ptr, data.data(), data.size()*sizeof(int), cudaMemcpyHostToDevice));

        GT_CUDA_CHECK(cudaMemcpy(data_out.data(), gpu_ptr, data.size()*sizeof(int), cudaMemcpyDeviceToHost));

        int i=0;
        for (auto x : data_out)
        {
            std::cout << x << " ";
            if (x!=i) 
                passed = false;
            ++i;
        }
        std::cout << std::endl;
    }
#endif

    
    EXPECT_TRUE(passed);
}
