#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "cg-actor.hpp"

using namespace caf;
using namespace caf::cuda;

void caf_main(actor_system& sys) {
  // Initialize the CUDA Manager
  manager_config config(true); 
  manager::init(sys, config);

  scoped_actor self{sys};

  // ===========================================================================
  // TEST 1: Standard SPD Matrix (Should converge normally)
  // ===========================================================================
  std::cout << "[INFO] --- Starting Test 1: Standard SPD Matrix ---" << std::endl;

  int n = 3;
  // Example: Solve Ax = b
  // A = [4, 1, 0; 1, 3, 0; 0, 0, 2] (Symmetric Positive Definite)
  std::vector<float> h_A = {4.0f, 1.0f, 0.0f, 
                            1.0f, 3.0f, 0.0f, 
                            0.0f, 0.0f, 2.0f};
  // b = [1, 2, 0.5]
  std::vector<float> h_b = {1.0f, 2.0f, 0.5f};
  // Initial guess x0 = [0, 0, 0]
  std::vector<float> h_x(n, 0.0f);

  // Transfer initial problem data to device memory
  command_runner<in<float>, in<float>, in_out<float>> setup_runner;
  auto results = setup_runner.transfer_memory(0, 0, create_in_arg(h_A), create_in_arg(h_b), create_in_out_arg(h_x));
  
  auto d_A = std::get<0>(results);
  auto d_b = std::get<1>(results);
  auto d_x = std::get<2>(results);

  auto solver = sys.spawn<cg_actor>(d_A, d_b, d_x, n, 1e-6f, 100, 0, 0);

  std::cout << "[INFO] Starting CG Solver..." << std::endl;
  
  // Call the actor to start solving
  self->mail(start_atom{}).send(solver);

  // Wait for the final solution vector (mem_ptr)
  self->receive(
    [&](mem_ptr<float> result_x) {
      auto host_x = result_x->copy_to_host();
      std::cout << "[SUCCESS] Solver finished. Result x: ";
      for (float val : host_x) std::cout << val << " ";
      std::cout << "\n" << std::endl;
    }
  );

  // ===========================================================================
  // TEST 2: Bad Input - Poorly Conditioned / Near-Singular Matrix
  // This test uses a matrix with a very high condition number.
  // The error reduction will be extremely slow, likely triggering the 
  // stagnation detection logic (threshold of 0.1% decrease over 15 iterations)
  // which will then trigger a Mathematical Restart [RECOVERY].
  // ===========================================================================
  std::cout << "[INFO] --- Starting Test 2: Bad Input (Poorly Conditioned) ---" << std::endl;

  int n2 = 2;
  // A = [1, 0; 0, 1e-7] -> Very high condition number
  std::vector<float> h_A2 = {1.0f, 0.0f, 
                             0.0f, 0.0000001f};
  // b = [1, 1]
  std::vector<float> h_b2 = {1.0f, 1.0f};
  std::vector<float> h_x2(n2, 0.0f);

  command_runner<in<float>, in<float>, in_out<float>> setup_runner2;
  auto results2 = setup_runner2.transfer_memory(0, 0, create_in_arg(h_A2), create_in_arg(h_b2), create_in_out_arg(h_x2));
  
  auto d_A2 = std::get<0>(results2);
  auto d_b2 = std::get<1>(results2);
  auto d_x2 = std::get<2>(results2);

  // Spawn with a high max_iter to allow time for stagnation detection to trigger
  auto solver2 = sys.spawn<cg_actor>(d_A2, d_b2, d_x2, n2, 1e-8f, 200, 0, 0);

  std::cout << "[INFO] Starting CG Solver with poor conditioning..." << std::endl;
  self->mail(start_atom{}).send(solver2);

  self->receive(
    [&](mem_ptr<float> result_x) {
      auto host_x = result_x->copy_to_host();
      std::cout << "[SUCCESS] Solver finished Test 2. Result x: ";
      for (float val : host_x) std::cout << val << " ";
      std::cout << "\n[INFO] Test 2 complete. Check logs for [RECOVERY] messages." << std::endl;
    },
    // Increase timeout for the poorly conditioned case
    after(std::chrono::seconds(20)) >> [] { std::cout << "[ERROR] Test 2 timed out!" << std::endl; }
  );

  // ===========================================================================
  // TEST 3: Stagnation Recovery (Hilbert Matrix)
  // Hilbert matrices are famously ill-conditioned. Even for small N, 
  // the ratio of max/min eigenvalues is huge, causing slow convergence.
  // This should trigger the stagnation detection logic (count >= 15)
  // because progress will be extremely slow, leading to a [RECOVERY] message.
  // ===========================================================================
  std::cout << "\n[INFO] --- Starting Test 3: Stagnation Recovery (Hilbert) ---" << std::endl;

  int n3 = 20; // Increased matrix size for more pronounced ill-conditioning
  std::vector<float> h_A3(n3 * n3);
  for (int i = 0; i < n3; ++i) {
    for (int j = 0; j < n3; ++j) {
      // Hilbert matrix element H_ij = 1 / (i + j + 1)
      h_A3[i * n3 + j] = 1.0f / (float)(i + j + 1);
    }
  }
  std::vector<float> h_b3(n3, 1.0f);
  std::vector<float> h_x3(n3, 0.0f);

  command_runner<in<float>, in<float>, in_out<float>> setup_runner3;
  auto results3 = setup_runner3.transfer_memory(0, 0, create_in_arg(h_A3), create_in_arg(h_b3), create_in_out_arg(h_x3));
  
  auto d_A3 = std::get<0>(results3);
  auto d_b3 = std::get<1>(results3);
  auto d_x3 = std::get<2>(results3);

  // Use a very high max_iter and tight tolerance to ensure we hit the 15-iteration stagnation threshold.
  auto solver3 = sys.spawn<cg_actor>(d_A3, d_b3, d_x3, n3, 1e-10f, 1000, 0, 0); // Increased max_iter to allow more iterations for stagnation

  std::cout << "[INFO] Starting CG Solver with Hilbert matrix..." << std::endl;
  self->mail(start_atom{}).send(solver3);

  self->receive(
    [&](mem_ptr<float> result_x) {
      auto host_x = result_x->copy_to_host();
      std::cout << "[SUCCESS] Solver finished Test 3. Result x: ";
      for (float val : host_x) std::cout << val << " ";
      std::cout << "\n[INFO] Test 3 complete. Check logs for [RECOVERY] messages." << std::endl;
    },
    after(std::chrono::seconds(120)) >> [] { std::cout << "[ERROR] Test 3 timed out!" << std::endl; } // Increased timeout for longer execution
  );

  // ===========================================================================
  // TEST 4: The "Narrow Valley" Matrix
  // A = [1, 1; 1, 1.00001]
  // This matrix is technically SPD, but the eigenvalues are roughly 2 and 0.000005.
  // This creates a extremely narrow "canyon" in the error surface.
  // Rounding errors will quickly make the residual drift from the true A*x - b.
  // ===========================================================================
  std::cout << "\n[INFO] --- Starting Test 4: Narrow Valley (Recovery Test) ---" << std::endl;

  int n4 = 2;
  float eps = 0.00001f;
  std::vector<float> h_A4 = {1.0f, 1.0f, 
                             1.0f, 1.0f + eps};
  std::vector<float> h_b4 = {2.0f, 2.0f + eps}; // Solution is exactly [1, 1]
  std::vector<float> h_x4(n4, 0.0f);

  command_runner<in<float>, in<float>, in_out<float>> setup_runner4;
  auto results4 = setup_runner4.transfer_memory(0, 0, create_in_arg(h_A4), create_in_arg(h_b4), create_in_out_arg(h_x4));
  
  auto d_A4 = std::get<0>(results4);
  auto d_b4 = std::get<1>(results4);
  auto d_x4 = std::get<2>(results4);

  // Tight tolerance to force many iterations
  auto solver4 = sys.spawn<cg_actor>(d_A4, d_b4, d_x4, n4, 1e-9f, 200, 0, 0);

  std::cout << "[INFO] Starting CG Solver with Narrow Valley matrix..." << std::endl;
  self->mail(start_atom{}).send(solver4);

  self->receive(
    [&](mem_ptr<float> result_x) {
      auto host_x = result_x->copy_to_host();
      std::cout << "[SUCCESS] Solver finished Test 4. Result x: ";
      for (float val : host_x) std::cout << val << " ";
      std::cout << "\n[INFO] Test 4 complete." << std::endl;
    },
    after(std::chrono::seconds(20)) >> [] { std::cout << "[ERROR] Test 4 timed out!" << std::endl; }
  );

  // ===========================================================================
  // TEST 5: Stress Test (Large Dense Matrix for Profiling)
  // Matrix Size: 16384 x 16384 (268M elements, ~1 GB)
  // This test is designed to saturate the GPU for a significant duration.
  // Use this to observe utilization, thermal throttling, and SM occupancy.
  // ===========================================================================
  std::cout << "\n[INFO] --- Starting Test 5: Stress Test (16384x16384) ---" << std::endl;

  int n5 = 16384;
  // 1D Laplacian (Poisson) matrix: 2 on diagonal, -1 on sub-diagonals.
  // Stored as a dense matrix to maximize GEMV computation.
  std::cout << "[INFO] Constructing 1GB matrix on host (this may take a moment)..." << std::endl;
  std::vector<float> h_A5(n5 * n5, 0.0f); 
  for (int i = 0; i < n5; ++i) {
    h_A5[i * n5 + i] = 2.0f;
    if (i > 0) h_A5[i * n5 + (i - 1)] = -1.0f;
    if (i < n5 - 1) h_A5[i * n5 + (i + 1)] = -1.0f;
  }
  std::vector<float> h_b5(n5, 1.0f);
  std::vector<float> h_x5(n5, 0.0f);

  std::cout << "[INFO] Transferring 1GB matrix to GPU..." << std::endl;
  command_runner<in<float>, in<float>, in_out<float>> setup_runner5;
  auto results5 = setup_runner5.transfer_memory(0, 0, create_in_arg(h_A5), create_in_arg(h_b5), create_in_out_arg(h_x5));
  
  auto d_A5 = std::get<0>(results5);
  auto d_b5 = std::get<1>(results5);
  auto d_x5 = std::get<2>(results5);
  
  // Run for 50,000 iterations with a near-zero tolerance to ensure sustained load.
  auto solver5 = sys.spawn<cg_actor>(d_A5, d_b5, d_x5, n5, 1e-18f, 50000, 0, 0);

  std::cout << "[INFO] Starting CG Solver stress test..." << std::endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  self->mail(start_atom{}).send(solver5);

  self->receive(
    [&](mem_ptr<float> result_x) {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
      std::cout << "[SUCCESS] Stress Test Finished in " << duration.count() << " seconds." << std::endl;
    },
    after(std::chrono::minutes(15)) >> [] { std::cout << "[ERROR] Test 5 timed out!" << std::endl; }
  );

  // ===========================================================================
  // TEST 6: Concurrent Stress Test (2 Actors, Same Device, Different Streams)
  // Both actors run the same 16384x16384 problem simultaneously.
  // This tests the framework's ability to handle high-load concurrency.
  // ===========================================================================
  std::cout << "\n[INFO] --- Starting Test 6: Concurrent Stress Test (2 Actors, Same Device, Diff Streams) ---" << std::endl;

  // Setup independent device vectors for Solver A (Stream 0) reusing host data from Test 5
  command_runner<in<float>, in_out<float>> vec_runner;
  auto res6a = vec_runner.transfer_memory(0, 0, create_in_arg(h_b5), create_in_out_arg(h_x5));
  auto d_b6a = std::get<0>(res6a);
  auto d_x6a = std::get<1>(res6a);

  // Setup independent device vectors for Solver B (Stream 1) reusing host data from Test 5
  auto res6b = vec_runner.transfer_memory(0, 1, create_in_arg(h_b5), create_in_out_arg(h_x5));
  auto d_b6b = std::get<0>(res6b);
  auto d_x6b = std::get<1>(res6b);

  auto solver6a = sys.spawn<cg_actor>(d_A5, d_b6a, d_x6a, n5, 1e-18f, 50000, 0, 0);
  auto solver6b = sys.spawn<cg_actor>(d_A5, d_b6b, d_x6b, n5, 1e-18f, 50000, 0, 1);

  std::cout << "[INFO] Launching both solvers concurrently..." << std::endl;
  auto start6 = std::chrono::high_resolution_clock::now();
  self->mail(start_atom{}).send(solver6a);
  self->mail(start_atom{}).send(solver6b);

  // Wait for both solvers to complete
  for (int i = 0; i < 2; ++i) {
    self->receive(
      [&](mem_ptr<float>) {
        std::cout << "[INFO] A solver in Test 6 has completed." << std::endl;
      },
      after(std::chrono::minutes(20)) >> [] { std::cout << "[ERROR] A solver in Test 6 timed out!" << std::endl; }
    );
  }
  auto end6 = std::chrono::high_resolution_clock::now();
  auto total_dur = std::chrono::duration_cast<std::chrono::seconds>(end6 - start6);
  std::cout << "[SUCCESS] Concurrent Stress Test Finished. Total wall-clock time: " << total_dur.count() << " seconds." << std::endl;

  manager::shutdown();
}

CAF_MAIN(id_block::cuda, id_block::cg_solver)