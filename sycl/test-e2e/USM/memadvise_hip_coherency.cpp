// RUN: %{build} -o %t1.out
// REQUIRES: hip_amd
// RUN: %{run} %t1.out

#include <sycl/sycl.hpp>

#include <chrono>
#include <iostream>

namespace kernels {
class SquareKrnl final {
public:
  SquareKrnl(int *ptr) : mPtr{ptr} {}

  void operator()(sycl::id<1>) const {
    // mPtr value squared here
    *mPtr = (*mPtr) * (*mPtr);
  }

private:
  int *mPtr;
};

class CoherencyTestKrnl final {
public:
  CoherencyTestKrnl(int *ptr) : mPtr{ptr} {}

  void operator()(sycl::id<1>) const {
    // mPtr was set to 1, now set it to 2
    auto atm = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                sycl::memory_scope::device>(mPtr[0]);
    atm.fetch_add(2);
    // ...
    int val = 3;
    while (true) {
      if (atm.compare_exchange_strong(val, val + 1))
        break;
    }
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

  bool coherent{false};

  int *ptr = sycl::malloc_shared<int>(1, q);

  // Hint that data coherency during simultaneous execution on
  // both host and device is not necessary
  constexpr int MemAdviseCoarseGrained{PI_MEM_ADVICE_HIP_SET_COARSE_GRAINED};
  q.mem_advise(ptr, sizeof(int), MemAdviseCoarseGrained);

  // TEST 1

  int number{9};
  int expected{number * number};

  // Call this routine TesCoherency function!
  *ptr = number;
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range{1}, kernels::SquareKrnl{ptr});
  });
  // Synchronise the underlying stream the work is run on before host access.
  q.wait();
  std::cout << *ptr << '\n';
  // Check if caches are flushed correctly and same memory is between devices.
  if (*ptr == expected) {
    coherent = true;
  } else {
    std::cout << "Coarse-grained mode coherency failed. Value = " << *ptr
              << '\n';
  }

  // TEST 2

  number = 1;
  expected = 4;

  *ptr = number;
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range{1}, kernels::CoherencyTestKrnl{ptr});
  });

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  while (std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::steady_clock::now() - start)
                 .count() < 3 &&
         *ptr == 2) {
  }          // wait till ptr is 2 from kernel or 3 seconds
  *ptr += 1; // increment it to 3

  // Synchronise the underlying stream the work is run on before host access.
  q.wait();
  std::cout << *ptr << '\n';
  // Check if caches are flushed correctly and same memory is between devices.
  if (*ptr == expected) {
    coherent &= true;
  } else {
    std::cout
        << "[CoherencyTestKrnl] Coarse-grained mode coherency failed. Value = "
        << *ptr << '\n';
  }

  // Cleanup
  sycl::free(ptr, q);

  // Check
  assert(coherent && "Coarse-grained mode coherency failed");

  return 0;
}
