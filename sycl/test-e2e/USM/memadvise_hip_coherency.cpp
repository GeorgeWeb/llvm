// RUN: %{build} -o %t1.out
// REQUIRES: hip_amd
// RUN: %{run} %t1.out

#include <sycl/sycl.hpp>

#include <chrono>
#include <iostream>

namespace kernels {
class SquareKrnl final {
  int *mPtr;

public:
  SquareKrnl(int *ptr) : mPtr{ptr} {}

  void operator()(sycl::id<1>) const {
    // mPtr value squared here
    *mPtr = (*mPtr) * (*mPtr);
  }
};

class CoherencyTestKrnl final {
  int *mPtr;

public:
  CoherencyTestKrnl(int *ptr) : mPtr{ptr} {}

  void operator()(sycl::id<1>) const {
    auto atm = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                sycl::memory_scope::device>(mPtr[0]);

    // mPtr was initialized to 1 by the host, now set it to 2.
    atm.fetch_add(1);
    int expected{3};
    while (true) {
      // spin until mPtr is 3, then change it to 4.
      if (atm.compare_exchange_strong(expected, 4))
        break;
    }
  }
};
} // namespace kernels

int main() {
  sycl::queue q{};
  sycl::device dev = q.get_device();
  sycl::context ctx = q.get_context();
  if (!dev.get_info<sycl::info::device::usm_shared_allocations>()) {
    std::cout << "Shared USM is not supported. Skipping test.\n";
    return 0;
  }

  bool coherent{false};

  int *ptr = sycl::malloc_shared<int>(1, q);

  // Hint that data coherency during simultaneous execution on
  // both host and device is not necessary
  constexpr int MemAdviseCoarseGrained{PI_MEM_ADVICE_HIP_SET_COARSE_GRAINED};
  q.mem_advise(ptr, sizeof(int), MemAdviseCoarseGrained);

  // Coherency test 1

  int init_val{9};
  int expected{init_val * init_val};

  *ptr = init_val;
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range{1}, kernels::SquareKrnl{ptr});
  });
  // Synchronise the underlying stream the work is run on before host access.
  q.wait();

  // Check if caches are flushed correctly and same memory is between devices.
  if (*ptr == expected) {
    coherent = true;
  } else {
    std::cerr << "[SquareKrnl] Coarse-grained mode coherency failed. Value = "
              << *ptr << '\n';
  }

  // Coherency test 2

  init_val = 1;
  expected = 4;

  *ptr = init_val;
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range{1}, kernels::CoherencyTestKrnl{ptr});
  });

  // wait until ptr is 2 from the kernel (or 3 seconds), then increment to 3.
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  while (std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::steady_clock::now() - start)
                 .count() < 3 &&
         *ptr == 2) {
  }
  *ptr += 1;

  // Synchronise the underlying stream the work is run on before host access.
  q.wait();

  // Check if caches are flushed correctly and same memory is between devices.
  if (*ptr == expected) {
    coherent &= true;
  } else {
    std::cerr
        << "[CoherencyTestKrnl] Coarse-grained mode coherency failed. Value = "
        << *ptr << '\n';
  }

  // Cleanup
  sycl::free(ptr, q);

  // Check
  assert(coherent && "Coarse-grained mode coherency failed");

  return 0;
}
