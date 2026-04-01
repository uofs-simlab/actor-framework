#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/component-actors/mmul_actor/mmul_actor.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <random>
#include "caf/actor_registry.hpp"
//#include <caf/atoms.hpp>



using namespace caf;
using namespace std::chrono_literals;

static const unsigned int RANDOM_SEED = 42;
// Written by the actor callback before self->quit(); read by run_mmul_test after
// await_all_actors_done() to surface the inner GPU-only total in the summary.
static double g_inner_total_ms = 0.0;


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

using command =
  caf::cuda::command_runner<>;

command mmul_command;

struct mmul_state {
};

//global output buffer meant to disclude it from timing
//the other benchmark test do not include its memory allocations in it 
//so its only fair that we do not either 
std::vector<int> matrixC;


caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_state>* self) {
  return {

    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB,
        int N) {

      using clock = std::chrono::steady_clock;
      using ms = std::chrono::duration<double, std::milli>;


      caf::cuda::manager& mgr = caf::cuda::manager::get();
      int device = 0;
      int stream = 1;

      auto program =
        mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

      auto t_total_start = clock::now();
      // -------------------------
      // create_in_arg A
      // -------------------------
      auto t_a_inarg_start = clock::now();

      auto inA = caf::cuda::create_in_arg(std::move(matrixA));

      auto t_a_inarg_end = clock::now();

      // -------------------------
      // transfer A
      // -------------------------
      auto t_a_transfer_start = clock::now();

      auto arg1 = mmul_command.transfer_memory(
          device,
          stream,
          std::move(inA));

      auto t_a_transfer_end = clock::now();

      // -------------------------
      // create_in_arg B
      // -------------------------
      auto t_b_inarg_start = clock::now();

      auto inB = caf::cuda::create_in_arg(std::move(matrixB));

      auto t_b_inarg_end = clock::now();

      // -------------------------
      // transfer B
      // -------------------------
      auto t_b_transfer_start = clock::now();

      auto arg2 = mmul_command.transfer_memory(
          device,
          stream,
          std::move(inB));

      auto t_b_transfer_end = clock::now();

      // -------------------------
      // spawn actor
      // -------------------------
      auto t_spawn_start = clock::now();

      caf::actor mmul_actor =
        self->spawn(caf::cuda::mmul_actor_fun<int>, program);

      auto t_spawn_end = clock::now();

      // -------------------------
      // request
      // -------------------------
      auto t_request_start = clock::now();

      self->mail(arg1, arg2, N, device, stream)
        .request(mmul_actor, std::chrono::seconds(30))
        .then(
          [=](caf::cuda::mem_ptr<int> dC) {

            auto t_response_received = clock::now();

            //std::vector<int> matrixC(N*N);
            // -------------------------
            // copy to host
            // -------------------------
            auto t_copy_start = clock::now();

            //std::vector<int> matrixC = dC->copy_to_host();

	    dC->copy_to_host(matrixC.data(),N*N);

            auto t_copy_end = clock::now();
            auto t_total_end = clock::now();

            // -------------------------
            // Print timings
            // -------------------------

            std::cout << "\n===== BENCHMARK RESULTS (N=" << N << ") =====\n";

            std::cout << "create_in_arg A: "
                      << ms(t_a_inarg_end - t_a_inarg_start).count()
                      << " ms\n";

            std::cout << "transfer A: "
                      << ms(t_a_transfer_end - t_a_transfer_start).count()
                      << " ms\n";

            std::cout << "create_in_arg B: "
                      << ms(t_b_inarg_end - t_b_inarg_start).count()
                      << " ms\n";

            std::cout << "transfer B: "
                      << ms(t_b_transfer_end - t_b_transfer_start).count()
                      << " ms\n";

            std::cout << "spawn actor: "
                      << ms(t_spawn_end - t_spawn_start).count()
                      << " ms\n";

            std::cout << "request → response latency: "
                      << ms(t_response_received - t_request_start).count()
                      << " ms\n";

            std::cout << "copy_to_host: "
                      << ms(t_copy_end - t_copy_start).count()
                      << " ms\n";

            std::cout << "TOTAL end-to-end: "
                      << ms(t_total_end - t_total_start).count()
                      << " ms\n";

            std::cout << "=============================================\n";

            self->quit();
          }
        );
    }

  };
}


caf::behavior mmul_actor_fun_2(caf::stateful_actor<mmul_state>* self) {
  return {

    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB,
        int N) {

      using clock = std::chrono::steady_clock;
      using ms = std::chrono::duration<double, std::milli>;


      caf::cuda::manager& mgr = caf::cuda::manager::get();
      int device = 0;
      int stream = 1;

      auto program =
        mgr.create_program_from_cubin("mmul.cubin", "matrixMul");

      auto t_total_start = clock::now();
      // -------------------------
      // create_in_arg A
      // -------------------------
      auto t_a_inarg_start = clock::now();

      auto inA = caf::cuda::create_in_arg(std::move(matrixA));

      auto t_a_inarg_end = clock::now();

      // -------------------------
      // transfer A
      // -------------------------
      auto t_a_transfer_start = clock::now();

      auto arg1 = mmul_command.transfer_memory(
          device,
          stream,
          std::move(inA));

      auto t_a_transfer_end = clock::now();

      // -------------------------
      // create_in_arg B
      // -------------------------
      auto t_b_inarg_start = clock::now();

      auto inB = caf::cuda::create_in_arg(std::move(matrixB));

      auto t_b_inarg_end = clock::now();

      // -------------------------
      // transfer B
      // -------------------------
      auto t_b_transfer_start = clock::now();

      auto arg2 = mmul_command.transfer_memory(
          device,
          stream,
          std::move(inB));

      auto t_b_transfer_end = clock::now();
  
      const int THREADS = 32;
      const int BLOCKS = (N + THREADS - 1) / THREADS;

      caf::cuda::nd_range dims(
        BLOCKS, BLOCKS, 1,
        THREADS, THREADS, 1);

      // -------------------------
      // request
      // -------------------------
      auto t_request_start = clock::now();

      caf::cuda::mmul_async_command<int> command;
      auto output = command.run_async(
		      program,dims,
		      1,
		      arg1,arg2,out<int>{N*N},in<int>{N});

       auto t_response_received = clock::now();

            //std::vector<int> matrixC(N*N);
            // -------------------------
            // copy to host
            // -------------------------
            auto t_copy_start = clock::now();

	    caf::cuda::mem_ptr<int> dC = std::get<2>(output);

            //std::vector<int> matrixC = dC->copy_to_host();

	    dC->copy_to_host(matrixC.data(),N*N);

            auto t_copy_end = clock::now();
            auto t_total_end = clock::now();

            // -------------------------
            // Print timings
            // -------------------------

            std::cout << "\n===== BENCHMARK RESULTS (N=" << N << ") =====\n";

            std::cout << "create_in_arg A: "
                      << ms(t_a_inarg_end - t_a_inarg_start).count()
                      << " ms\n";

            std::cout << "transfer A: "
                      << ms(t_a_transfer_end - t_a_transfer_start).count()
                      << " ms\n";

            std::cout << "create_in_arg B: "
                      << ms(t_b_inarg_end - t_b_inarg_start).count()
                      << " ms\n";

            std::cout << "transfer B: "
                      << ms(t_b_transfer_end - t_b_transfer_start).count()
                      << " ms\n";

                       std::cout << "request → response latency: "
                      << ms(t_response_received - t_request_start).count()
                      << " ms\n";

            std::cout << "copy_to_host: "
                      << ms(t_copy_end - t_copy_start).count()
                      << " ms\n";

            double inner_total = ms(t_total_end - t_total_start).count();
            g_inner_total_ms = inner_total;
            std::cout << "TOTAL end-to-end: " << inner_total << " ms\n";

            std::cout << "=============================================\n";

           self->quit();
    }

  };
}


double run_mmul_test(caf::actor_system& sys, int matrix_size) {
  // ------------------------------------
  // Start timing
  // ------------------------------------
  auto start = std::chrono::steady_clock::now();

  // Use seeded random data to match main_cuda_native
  std::mt19937 rng(RANDOM_SEED);
  std::uniform_int_distribution<int> dist(1, 10);
  std::vector<int> matrixA(matrix_size * matrix_size);
  std::vector<int> matrixB(matrix_size * matrix_size);
  for (auto& v : matrixA) v = dist(rng);
  for (auto& v : matrixB) v = dist(rng);

  matrixC.resize(matrix_size * matrix_size);

  caf::actor a = sys.spawn(mmul_actor_fun_2);
  anon_mail(matrixA, matrixB, matrix_size).send(a);

  // Wait for all actors to finish
  sys.await_all_actors_done();

  // ------------------------------------
  // Stop timing
  // ------------------------------------
  auto end = std::chrono::steady_clock::now();
  double duration_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "[MMUL TEST] matrix_size=" << matrix_size
            << ", time=" << duration_ms << " ms\n";
  return duration_ms;
}


void caf_main(caf::actor_system& sys) {
  std::vector<int> sizes = {1000, 2000, 4000, 8000, 16000};
  std::vector<std::pair<int,double>> outer_results;
  std::vector<std::pair<int,double>> inner_results;

  // Init CUDA manager once for all tests (same as native which creates context once)
  caf::cuda::manager::init(sys);

  // Warmup: small run to prime CUDA lazy-init / cubin load before timed tests
  {
    int W = 64;
    std::vector<int> wa(W * W, 1), wb(W * W, 1);
    matrixC.resize(W * W);
    caf::actor warmup = sys.spawn(mmul_actor_fun_2);
    anon_mail(wa, wb, W).send(warmup);
    sys.await_all_actors_done();
    std::cout << "--- warmup complete ---\n";
  }

  for (int N : sizes) {
    double t_outer = run_mmul_test(sys, N);
    outer_results.emplace_back(N, t_outer);
    inner_results.emplace_back(N, g_inner_total_ms);
  }

  caf::cuda::manager::shutdown();

  // Summary: outer time = full actor overhead (spawn+msg+await)
  //          inner time = GPU-only time comparable directly to main_cuda_native TOTAL
  std::cout << "\nMatrix size : outer time (ms) : inner GPU time (ms)\n";
  for (size_t i = 0; i < outer_results.size(); ++i) {
    std::cout << outer_results[i].first
              << " : " << outer_results[i].second
              << " : " << inner_results[i].second << "\n";
  }

}




CAF_MAIN()
