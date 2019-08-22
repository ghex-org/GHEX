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
#include <gridtools/common/host_device.hpp>
#endif

#include <type_traits>
#include <vector>
#include <iostream>

#include <gtest/gtest.h>
#include "gtest_main_gpu.cpp"

#include <boost/mpi/communicator.hpp>

#include "../include/devices.hpp"

#ifdef __CUDACC__
template<typename T, int N>
struct gpu_array
{
    GT_HOST gpu_array(std::array<T,N>&& arr)
    : gpu_array(&arr[0]) {}

    GT_HOST_DEVICE gpu_array(T* ptr)
    {
        for (int i=0; i<N; ++i) m_data[i] = ptr[i];
    }

    GT_HOST_DEVICE gpu_array(const gpu_array& other) = default;
    GT_HOST_DEVICE gpu_array(gpu_array&&) = default;

    GT_HOST_DEVICE T operator[](int i) const
    {
        return m_data[i];
    }

    GT_HOST_DEVICE T& operator[](int i)
    {
        return m_data[i];
    }

    T m_data[N];
};
#endif

#ifdef __CUDACC__
template<typename T>
__global__ void test_stencil(const T* in, T* out, int size, gpu_array<int,3> arr)
{
    //const auto index = blockIdx.x*blockDim.x + threadIdx.x;
    const auto index = threadIdx.x;
    if (index < arr[0])
    {
        out[index] = in[index];
    }
}

template<typename T, int N>
__global__ void serialize(const T* field, T* buffer, gpu_array<int,3> first)
{
    //const auto index = blockIdx.x*blockDim.x + threadIdx.x;
    const auto index = threadIdx.x;
    if (index < arr[0])
    {
        out[index] = in[index];
    }
}
#endif


TEST(gpu, allocate)
{
    bool passed = true;

    boost::mpi::communicator world;

#ifdef __CUDACC__

    int num_devices_per_node;
    cudaGetDeviceCount(&num_devices_per_node);

    MPI_Comm raw_local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world.rank(), MPI_INFO_NULL, &raw_local_comm);
    boost::mpi::communicator local_comm(raw_local_comm, boost::mpi::comm_take_ownership);

    if (local_comm.rank()<num_devices_per_node)
    {
        std::cout << "I am rank " << world.rank() << " and I own GPU " 
        << (world.rank()/local_comm.size())*num_devices_per_node + local_comm.rank() << std::endl;
        GT_CUDA_CHECK(cudaSetDevice(local_comm.rank()));
        std::cout << "gpu is invoked!" << std::endl;

        std::vector<int> data = {0,1,2,3,4,5,6};
        int* gpu_ptr;
        int* gpu_ptr2;
        //GT_CUDA_CHECK(cudaMalloc(&gpu_ptr, data.size()*sizeof(int)));
        //GT_CUDA_CHECK(cudaMalloc(&gpu_ptr2, data.size()*sizeof(int)));
        GT_CUDA_CHECK(cudaMalloc((void**)&gpu_ptr2, data.size()*sizeof(int)));
        GT_CUDA_CHECK(cudaMalloc((void**)&gpu_ptr, data.size()*sizeof(int)));

        auto gpu_vec_1 = gridtools::device::gpu::make_vector<int>();
        auto gpu_vec_2 = gridtools::device::gpu::make_vector<int>();
        gpu_vec_1.resize(data.size());
        gpu_vec_2.resize(data.size());
        std::vector<int> data_out(data.size());
        

        if (world.rank()==0)
        {
            //GT_CUDA_CHECK(cudaMemcpy(gpu_ptr, data.data(), data.size()*sizeof(int), cudaMemcpyHostToDevice));
            //cudaFree(gpu_ptr); 
            //GT_CUDA_CHECK(cudaMalloc((void**)&gpu_ptr, data.size()*sizeof(int)*2));
            
            //GT_CUDA_CHECK(cudaMemcpy(gpu_ptr, data.data(), data.size()*sizeof(int), cudaMemcpyHostToDevice));
            GT_CUDA_CHECK(cudaMemcpy(gpu_vec_1.data(), data.data(), data.size()*sizeof(int), cudaMemcpyHostToDevice));

            //MPI_Send(gpu_ptr, data.size()*sizeof(int), MPI_BYTE, 12, 0, world);
            MPI_Send(gpu_vec_1.data(), data.size()*sizeof(int), MPI_BYTE, 12, 0, world);

            gpu_array<int,3> arr{{(int)data.size(), 0, 0}};
            //test_stencil<<<1,512>>>(gpu_ptr, gpu_ptr2, data.size());
            test_stencil<<<1,512>>>(gpu_vec_1.data(), gpu_vec_2.data(), data.size(), arr);

            //GT_CUDA_CHECK(cudaMemcpy(data_out.data(), gpu_ptr2, data.size()*sizeof(int), cudaMemcpyDeviceToHost));
            GT_CUDA_CHECK(cudaMemcpy(data_out.data(), gpu_vec_2.data(), data.size()*sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (world.rank()==12)
        {
            MPI_Status status;
            MPI_Recv(gpu_ptr, data.size()*sizeof(int), MPI_BYTE,  0, 0, world, &status);
            GT_CUDA_CHECK(cudaMemcpy(data_out.data(), gpu_ptr, data.size()*sizeof(int), cudaMemcpyDeviceToHost));
        }
        else
        {
            GT_CUDA_CHECK(cudaMemcpy(gpu_ptr, data.data(), data.size()*sizeof(int), cudaMemcpyHostToDevice));
            GT_CUDA_CHECK(cudaMemcpy(data_out.data(), gpu_ptr, data.size()*sizeof(int), cudaMemcpyDeviceToHost));
        }

        cudaFree(gpu_ptr); 
        cudaFree(gpu_ptr2);

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
