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

  // Spawn the Conjugate Gradient Actor
  auto solver = sys.spawn<cg_actor>(d_A, d_b, d_x, n, 1e-6f, 100);

  scoped_actor self{sys};
  std::cout << "[INFO] Starting CG Solver..." << std::endl;
  
  // Call the actor to start solving
  self->mail(start_atom{}).send(solver);

  // Wait for the final solution vector (mem_ptr)
  self->receive(
    [&](mem_ptr<float> result_x) {
      auto host_x = result_x->copy_to_host();
      std::cout << "[SUCCESS] Solver finished. Result x: ";
      for (float val : host_x) std::cout << val << " ";
      std::cout << std::endl;
    }
  );

  manager::shutdown();
}

CAF_MAIN(id_block::cuda, id_block::cg_solver)