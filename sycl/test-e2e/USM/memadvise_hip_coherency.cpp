// RUN: %{build} -o %t1.out
// REQUIRES: hip_amd
// RUN: %{run} %t1.out

#include <iostream>
#include <sycl/sycl.hpp>

namespace kernels {
class SquareKrnl final {
public:
  SquareKrnl(int *ptr) : mPtr(ptr) {}

  void operator()(sycl::id<1>) const {
    // mPtr value squared here
    *mPtr = (*mPtr) * (*mPtr);
  }

private:
  int *mPtr;
};
} // namespace kernels

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();
  sycl::context ctx = q.get_context();
  if (!dev.get_info<sycl::info::device::usm_shared_allocations>()) {
    std::cout << "Shared USM is not supported. Skipping test.\n";
    return 0;
  }

  int *ptr = sycl::malloc_shared<int>(1, q);

  // Hint that data coherency during simultaneous execution on
  // both host and device is not necessary
  constexpr int MemAdviseCoarseGrained = PI_MEM_ADVICE_HIP_SET_COARSE_GRAINED;
  q.mem_advise(ptr, sizeof(int), MemAdviseCoarseGrained);

  // Call this routine TesCoherency function!
  constexpr int number = 9;
  constexpr int expected = 81;
  *ptr = number;
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range{1}, kernels::SquareKrnl(ptr));
  });
  // Synchronise the underlying stream the work is run on before host access.
  q.wait();
  // Check if caches are flushed correctly and same memory is between devices.
  if (*ptr != expected) {
    std::cout << "Coarse-grained mode coherency failed. Value = " << *ptr
              << '\n';
    return 1;
  }

  sycl::free(ptr, q);
  return 0;
}
