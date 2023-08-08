// RUN: %{build} -o %t1.out
// REQUIRES: hip_amd
// RUN: %{run} %t1.out

//==---------------- memadvise_cuda.cpp ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/context.hpp"
#include "sycl/detail/pi.h"
#include "sycl/device.hpp"
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();
  sycl::context ctx = q.get_context();
  if (!dev.get_info<sycl::info::device::usm_shared_allocations>()) {
    std::cout << "Shared USM is not supported. Skipping test." << std::endl;
    return 0;
  }

  constexpr size_t size = 100;
  void *ptr = sycl::malloc_shared(size, dev, ctx);
  if (ptr == nullptr) {
    std::cout << "Allocation failed!" << std::endl;
    return -1;
  }

  // NOTE: PI_MEM_ADVICE_CUDA_* advice values are mapped to the HIP backend too.
  std::vector<int> valid_advices{
      PI_MEM_ADVICE_CUDA_SET_READ_MOSTLY,
      PI_MEM_ADVICE_CUDA_UNSET_READ_MOSTLY,
      PI_MEM_ADVICE_CUDA_SET_PREFERRED_LOCATION,
      PI_MEM_ADVICE_CUDA_UNSET_PREFERRED_LOCATION,
      PI_MEM_ADVICE_CUDA_SET_ACCESSED_BY,
      PI_MEM_ADVICE_CUDA_UNSET_ACCESSED_BY,
      PI_MEM_ADVICE_CUDA_SET_PREFERRED_LOCATION_HOST,
      PI_MEM_ADVICE_CUDA_UNSET_PREFERRED_LOCATION_HOST,
      PI_MEM_ADVICE_CUDA_SET_ACCESSED_BY_HOST,
      PI_MEM_ADVICE_CUDA_UNSET_ACCESSED_BY_HOST,
      PI_MEM_ADVICE_HIP_SET_COARSE_GRAINED,
      PI_MEM_ADVICE_HIP_UNSET_COARSE_GRAINED,
  };
  for (int advice : valid_advices) {
    q.mem_advise(ptr, size, advice);
  }

  q.wait_and_throw();
  std::cout << "Test passed." << std::endl;
  return 0;
}
