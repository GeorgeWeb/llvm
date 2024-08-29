//==---------------- reduction.cpp - SYCL reduction ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "detail/context_impl.hpp"
#include "sycl/backend_types.hpp"
#include "detail/kernel_bundle_impl.hpp"
#include "sycl/context.hpp"
#include "sycl/info/info_desc.hpp"
#include "sycl/kernel_bundle.hpp"
#include "sycl/kernel_bundle_enums.hpp"
#include <cstdint>
#include <detail/config.hpp>
#include <detail/memory_manager.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/reduction.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

__SYCL_EXPORT kernel reduGetKernelExec(std::shared_ptr<queue_impl> Queue,
                                       std::string_view KernelName) {
  std::cout << "Kernel Name: " << KernelName.data() << '\n';
  context Ctx = Queue->get_context();
  device Device = Queue->get_device();
  auto KernelId = get_kernel_id_impl(KernelName);
  static constexpr bundle_state State{bundle_state::executable};
  auto KernelBundleImpl = get_kernel_bundle_impl(Ctx, {Device}, State);
  auto Kernel = KernelBundleImpl->get_kernel(KernelId, KernelBundleImpl);
  return Kernel;
}

// TODO: The algorithm of choosing the work-group size is definitely
// imperfect now and can be improved.
__SYCL_EXPORT size_t reduComputeWGSize(size_t NWorkItems, size_t MaxWGSize,
                                       size_t &NWorkGroups) {
  size_t WGSize = MaxWGSize;
  if (NWorkItems <= WGSize) {
    NWorkGroups = 1;
    WGSize = NWorkItems;
  } else {
    NWorkGroups = NWorkItems / WGSize;
    size_t Rem = NWorkItems % WGSize;
    if (Rem != 0) {
      // Let's suppose MaxWGSize = 128 and NWorkItems = (128+32).
      // It seems better to have 5 groups 32 work-items each than 2 groups with
      // 128 work-items in the 1st group and 32 work-items in the 2nd group.
      size_t NWorkGroupsAlt = NWorkItems / Rem;
      size_t RemAlt = NWorkItems % Rem;
      if (RemAlt == 0 && NWorkGroupsAlt <= MaxWGSize) {
        // Choose smaller uniform work-groups.
        // The condition 'NWorkGroupsAlt <= MaxWGSize' was checked to ensure
        // that choosing smaller groups will not cause the need in additional
        // invocations of the kernel.
        NWorkGroups = NWorkGroupsAlt;
        WGSize = Rem;
      } else {
        // Add 1 more group to process the remaining elements and proceed
        // with bigger non-uniform work-groups
        NWorkGroups++;
      }
    }
  }
  std::cout << "Maximum WorkGroup Size: " << MaxWGSize << '\n';
  std::cout << "Compute WorkGroup Size: " << WGSize << '\n';
  return WGSize;
}

__SYCL_EXPORT bool
reduShouldUseKernelBundle(std::shared_ptr<queue_impl> Queue) {
  if (!Queue)
    return false;

  const device Device = Queue->get_device();
  const backend Backend = Device.get_backend();
  return SYCLConfig<SYCL_REDUCTION_ENABLE_USE_KERNEL_BUNDLES>::get(Backend);
};

// Returns the estimated number of physical threads on the device associated
// with the given queue.
__SYCL_EXPORT uint32_t reduGetMaxNumConcurrentWorkGroups(
    std::shared_ptr<sycl::detail::queue_impl> Queue) {
  // TODO: Graphs extension explicit API uses a handler with no queue attached,
  // so return some value here. In the future we should have access to the
  // device so can remove this.
  //
  // The 8 value was chosen as the hardcoded value as it is the returned
  // value for sycl::info::device::max_compute_units on
  // Intel HD Graphics devices used as a L0 backend during development.
  if (Queue == nullptr) {
    return 8;
  }
  device Dev = Queue->get_device();
  uint32_t NumThreads = Dev.get_info<sycl::info::device::max_compute_units>();
  // TODO: The heuristics here require additional tuning for various devices
  // and vendors. Also, it would be better to check vendor/generation/etc.
  if (Dev.is_gpu() && Dev.get_info<sycl::info::device::host_unified_memory>())
    NumThreads *= 8;
  return NumThreads;
}

/// Check if the (unsigned) value of N is a power-of-two.
inline bool IsPowerOfTwo(size_t N) noexcept {
  return (N & (N - 1)) == 0;
}

__SYCL_EXPORT size_t
reduGetMaxWGSize(std::shared_ptr<sycl::detail::queue_impl> Queue,
                 size_t LocalMemBytesPerWorkItem) {
  std::cout << "GetMaxWGSize from device query.\n";
  device Dev = Queue->get_device();
  size_t MaxWGSize = Dev.get_info<sycl::info::device::max_work_group_size>();

  size_t WGSizePerMem = MaxWGSize * 2;
  size_t WGSize = MaxWGSize;
  if (LocalMemBytesPerWorkItem != 0) {
    size_t MemSize = Dev.get_info<sycl::info::device::local_mem_size>();
    WGSizePerMem = MemSize / LocalMemBytesPerWorkItem;

    // If the work group size is NOT power of two, then an additional element
    // in local memory is needed for the reduction algorithm and thus the real
    // work-group size requirement per available memory is stricter.
    if ((WGSizePerMem & (WGSizePerMem - 1)) != 0)
      WGSizePerMem--;
    WGSize = (std::min)(WGSizePerMem, WGSize);
  }
  // TODO: This is a temporary workaround for a big problem of detecting
  // the maximal usable work-group size. The detection method used above
  // is based on maximal work-group size possible on the device is too risky
  // as may return too big value. Even though it also tries using the memory
  // factor into consideration, it is too rough estimation. For example,
  // if (WGSize * LocalMemBytesPerWorkItem) is equal to local_mem_size, then
  // the reduction local accessor takes all available local memory for it needs
  // not leaving any local memory for other kernel needs (barriers,
  // builtin calls, etc), which often leads to crashes with OUT_OF_RESOURCES
  // error, or in even worse cases it may cause silent writes/clobbers of
  // the local memory assigned to one work-group by code in another work-group.
  // It seems the only good solution for this work-group detection problem is
  // kernel precompilation and querying the kernel properties.
  if (WGSize >= 4 && WGSizePerMem < MaxWGSize * 2) {
    // Let's return a twice smaller number, but... do that only if the kernel
    // is limited by memory.
    WGSize /= 2;
  }

  ///*
  // Terrible consrevative workaround without access to kernel properties.
  std::cout << "WGSize before conservative regs workaround: " << WGSize << '\n';
  size_t NewWGSize{WGSize};
  if (Dev.get_backend() == backend::ext_oneapi_cuda) {
    namespace codeplay = sycl::ext::codeplay;
    const uint32_t MaxRegsPerWG = Dev.get_info<
        codeplay::experimental::info::device::max_registers_per_work_group>();
    // Assumes using max number of 32-bit registers per thread in CUDA (255).
    // see: link-to-cuda-cap-table
    constexpr uint8_t MaxRegsPerWI{255};
    while (NewWGSize * MaxRegsPerWI > MaxRegsPerWG || !IsPowerOfTwo(NewWGSize))
      NewWGSize--;
  }
  std::cout << "WGSize after conservative regs workaround: " << NewWGSize << '\n';
  //*/
  return WGSize;
}

__SYCL_EXPORT size_t
reduGetMaxWGSize(std::shared_ptr<sycl::detail::queue_impl> Queue,
                 const sycl::kernel& Kernel,
                 size_t LocalMemBytesPerWorkItem = 0) {
  std::cout << "GetMaxWGSize from kernel query.\n";
  device Device = Queue->get_device();
  size_t MaxWGSize =
      Kernel.get_info<info::kernel_device_specific::work_group_size>(Device);

  // Handle case where the backend does not have an implementation of the query.
  if (MaxWGSize == 0)
    return reduGetMaxWGSize(Queue, LocalMemBytesPerWorkItem);

  std::cout << "[reduGetMaxWGSize] MaxWGSize: " << MaxWGSize << '\n';
  return MaxWGSize;
}

__SYCL_EXPORT size_t reduGetPreferredWGSize(std::shared_ptr<queue_impl> &Queue,
                                            size_t LocalMemBytesPerWorkItem) {
  std::cout << "reduGetPreferredWGSize check\n";
  // TODO: Graphs extension explicit API uses a handler with a null queue to
  // process CGFs, in future we should have access to the device so we can
  // correctly calculate this.
  //
  // The 32 value was chosen as the hardcoded value as it is the returned
  // value for SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE on
  // Intel HD Graphics devices used as a L0 backend during development.
  if (Queue == nullptr) {
    return 32;
  }
  device Dev = Queue->get_device();

  // The maximum WGSize returned by CPU devices is very large and does not
  // help the reduction implementation: since all work associated with a
  // work-group is typically assigned to one CPU thread, selecting a large
  // work-group size unnecessarily increases the number of accumulators.
  // The default of 16 was chosen based on empirical benchmarking results;
  // an environment variable is provided to allow users to override this
  // behavior.
  using PrefWGConfig = sycl::detail::SYCLConfig<
      sycl::detail::SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE>;
  if (Dev.is_cpu()) {
    size_t CPUMaxWGSize = PrefWGConfig::get(sycl::info::device_type::cpu);
    if (CPUMaxWGSize == 0)
      return 16;
    size_t DevMaxWGSize =
        Dev.get_info<sycl::info::device::max_work_group_size>();
    return std::min(CPUMaxWGSize, DevMaxWGSize);
  }

  // If the user has specified an explicit preferred work-group size we use
  // that.
  if (Dev.is_gpu() && PrefWGConfig::get(sycl::info::device_type::gpu)) {
    size_t DevMaxWGSize =
        Dev.get_info<sycl::info::device::max_work_group_size>();
    return std::min(PrefWGConfig::get(sycl::info::device_type::gpu),
                    DevMaxWGSize);
  }

  if (Dev.is_accelerator() &&
      PrefWGConfig::get(sycl::info::device_type::accelerator)) {
    size_t DevMaxWGSize =
        Dev.get_info<sycl::info::device::max_work_group_size>();
    return std::min(PrefWGConfig::get(sycl::info::device_type::accelerator),
                    DevMaxWGSize);
  }

  // Use the maximum work-group size otherwise.
  return reduGetMaxWGSize(Queue, LocalMemBytesPerWorkItem);
}

__SYCL_EXPORT void
addCounterInit(handler &CGH, std::shared_ptr<sycl::detail::queue_impl> &Queue,
               std::shared_ptr<int> &Counter) {
  auto EventImpl = std::make_shared<detail::event_impl>(Queue);
  EventImpl->setContextImpl(detail::getSyclObjImpl(Queue->get_context()));
  EventImpl->setStateIncomplete();
  MemoryManager::fill_usm(Counter.get(), Queue, sizeof(int), {0}, {},
                          &EventImpl->getHandleRef(), EventImpl);
  CGH.depends_on(createSyclObjFromImpl<event>(EventImpl));
}

} // namespace detail
} // namespace _V1
} // namespace sycl
