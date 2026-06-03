/* 
 * Modern Actor Facade Example: Matrix Multiplication
 * 
 * This file demonstrates the three primary ways to interact with the caf::cuda::actor_facade.
 * The facade is fully asynchronous and pushes results back to the requester's mailbox.
 * 
 * Requirements: Run compile_kernels.sh to generate mmul.cubin before running.
 */
#include <caf/all.hpp>  // Includes most CAF essentials
#include "caf/cuda/actor_facade.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/all.hpp"
#include <caf/type_id.hpp>
#include "caf/detail/test.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace caf;
using namespace std::chrono_literals;
using namespace caf::cuda;

// Verification function for matrix multiplication on the CPU
void serial_matrix_multiply(const std::vector<int>& a,
                            const std::vector<int>& b,
                            std::vector<int>& c,
                            int N) {
  

 for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int sum = 0;
      for (int k = 0; k < N; ++k) {
        sum += a[i * N + k] * b[k * N + j];
      }
      c[i * N + j] = sum;
    }
  }
}

// Common state for all testers to hold input vectors and output buffer for mapping
struct mmul_tester_state {
  std::vector<int> A;
  std::vector<int> B;
  std::vector<int> host_buffer; // Used by mapping_tester
};

// ---------------------------------------------------------------------------
// Case 1: Standard Vector Results
// The facade copies results from Device to Host and sends std::vector<T>.
// ---------------------------------------------------------------------------
caf::behavior standard_tester(caf::stateful_actor<mmul_tester_state>* self, caf::actor facade,
                              int N, std::vector<int> ref, int launch_id) {
  std::cout << "[Tester] Launching Standard Facade Test...\n";

  self->state().A.assign(N * N, 1); // Store A in actor state
  self->state().B.assign(N * N, 2); // Store B in actor state
  auto arg1 = create_in_arg(self->state().A);
  auto arg2 = create_in_arg(self->state().B);
  auto arg3 = create_out_arg_with_size<int>(N * N);
  auto arg4 = create_in_arg(N);

  // Send work. We use 'mail' to provide type-safe arguments.
  self->mail(arg1, arg2, arg3, arg4).send(facade);

  return {
    // Signature: (int reply_id, int index, std::vector<T> data)
    [=](int r_id, int index, std::vector<int> data) {
      if (r_id == launch_id) {
        std::cout << "[Standard] Received result for index " << index << "\n";
        if (index == 2) { // Matrix C
          bool match = (data == ref);
          std::cout << (match ? "[PASS]" : "[FAIL]") << " Standard Vector Match\n";
        }
        self->quit();
      }
    }
  };
}

// ---------------------------------------------------------------------------
// Case 2: Memory Pointer Results
// The facade returns raw mem_ptr<T> handles instead of copying data to host.
// ---------------------------------------------------------------------------
caf::behavior memptr_tester(caf::stateful_actor<mmul_tester_state>* self, caf::actor facade,
                            int N, std::vector<int> ref, int launch_id) {
  std::cout << "[Tester] Launching MemPtr Facade Test...\n";

  self->state().A.assign(N * N, 1);
  self->state().B.assign(N * N, 2);
  auto arg1 = create_in_arg(self->state().A);
  auto arg2 = create_in_arg(self->state().B);
  auto arg3 = create_out_arg_with_size<int>(N * N);
  auto arg4 = create_in_arg(N);

  // By prepending return_mem_ptr_atom, the facade returns handles immediately 
  // after kernel launch, allowing manual control over Device-to-Host moves.
  self->mail(return_mem_ptr_atom_v, arg1, arg2, arg3, arg4).send(facade);

  return {
    // Signature: (int reply_id, mem_ptr<T>...)
    [=](int r_id, mem_ptr<int> pA, mem_ptr<int> pB, mem_ptr<int> pC, mem_ptr<int> pN) {
      if (r_id == launch_id) {
        std::cout << "[MemPtr] Received device memory handles.\n";
        
        // Copy matrix C back to host manually.
        std::vector<int> data = pC->copy_to_host();
        
        bool match = (data == ref);
        std::cout << (match ? "[PASS]" : "[FAIL]") << " MemPtr Data Match\n";
        self->quit();
      }
    }
  };
}

// ---------------------------------------------------------------------------
// Case 3: Output Mapping
// The facade copies results directly into a pre-allocated host buffer.
// ---------------------------------------------------------------------------
caf::behavior mapping_tester(caf::stateful_actor<mmul_tester_state>* self,
                             caf::actor facade, int N, std::vector<int> ref, 
                             int launch_id) {
  std::cout << "[Tester] Launching Output Mapping Test...\n";
  
  self->state().A.assign(N * N, 1); // Store A in actor state
  self->state().B.assign(N * N, 2); // Store B in actor state
  self->state().host_buffer.assign(N * N, 0);

  auto arg1 = create_in_arg(self->state().A);
  auto arg2 = create_in_arg(self->state().B);
  auto arg3 = create_out_arg_with_size<int>(N * N);
  auto arg4 = create_in_arg(N);

  // Define the mapping: tell the facade to put index 2 into our local vector.
  std::vector<output_mapping> mappings = {
    {2, self->state().host_buffer.data(), self->state().host_buffer.size()}
  };

  self->mail(mappings, arg1, arg2, arg3, arg4).send(facade);

  return {
    // Signature: (int reply_id, int index)
    [=](int r_id, int index) {
      if (r_id == launch_id) {
        std::cout << "[Mapping] Facade notified completion for index " << index << "\n";
        if (index == 2) {
          bool match = (self->state().host_buffer == ref);
          std::cout << (match ? "[PASS]" : "[FAIL]") << " Mapped Buffer Match\n";
        }
        self->quit();
      }
    }
  };
}

void caf_main(caf::actor_system& sys) {
  // 1. Initialize the CUDA Manager
  manager::init(sys);
  auto& mgr = manager::get();

  // 2. Setup Kernel Dimensions and Metadata
  int N = 128;
  int threads = 32;
  int blocks = (N + threads - 1) / threads;
  nd_range dims(blocks, blocks, 1, threads, threads, 1);

  // 3. Create Reference Data on CPU
  std::vector<int> h_a(N * N, 1), h_b(N * N, 2), h_ref(N * N, 0);
  serial_matrix_multiply(h_a, h_b, h_ref, N);

  // 4. Spawn the Facade
  // We pass the argument tags (in/out) so the facade knows the kernel signature.
  int my_reply_id = 0;
  auto mmul_facade = mgr.spawnFromCUBIN(
    "../mmul.cubin", "matrixMul", dims, 
    in<int>{}, in<int>{}, out<int>{}, in<int>{}
  );

  // 5. Run the interaction tests
  sys.spawn(standard_tester, mmul_facade, N, h_ref, my_reply_id);
  sys.spawn(memptr_tester, mmul_facade, N, h_ref, my_reply_id);
  sys.spawn(mapping_tester, mmul_facade, N, h_ref, my_reply_id);
}

CAF_MAIN()
