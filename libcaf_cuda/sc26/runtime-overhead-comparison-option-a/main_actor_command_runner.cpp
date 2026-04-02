// main_actor_command_runner.cpp
// Option A: Serialised latency — command-runner variant unchanged from the
// corrected runtime-overhead-comparison test. The command_runner.run() call is
// already synchronous (blocks until GPU completes), so no structural change is
// needed. This file exists to pair with the Option A native variant which now
// also serialises after every async op.
#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>

using namespace caf;
static const unsigned int RANDOM_SEED = 42;
using namespace std::chrono_literals;
using mmul_command = caf::cuda::command_runner<
    in<int>,  // matrix A
    in<int>,  // matrix B
    out<int>, // matrix C
    in<int>   // matrix size N
>;


class MatMult {
  event_based_actor* self_;
  std::vector<int> A_;
  std::vector<int> B_;
  caf::cuda::program_ptr program_;
  using clock = std::chrono::steady_clock;
  clock::time_point start_;

public:
  MatMult(event_based_actor* self, int N) : self_(self) {
    std::mt19937 rng(RANDOM_SEED);
    std::uniform_int_distribution<int> dist(1, 10);

    A_.resize(N * N);
    B_.resize(N * N);

    for (auto& v : A_) v = dist(rng);
    for (auto& v : B_) v = dist(rng);

    // Pre-load cubin outside the timed window
    program_ = self_->system().cuda_manager().create_program_from_cubin(
        "mmul.cubin", "matrixMul");
  };

  behavior make_behavior() {
    return {
      [this](int N) {
        start_ = clock::now();

        int THREADS = 32;
        int BLOCKS = (N + THREADS - 1) / THREADS;

        caf::cuda::nd_range dim(BLOCKS,
                                BLOCKS,
                                1,
                                THREADS,
                                THREADS,
                                1);
        
        auto arg1 = caf::cuda::create_in_arg(std::move(A_));
        auto arg2 = caf::cuda::create_in_arg(std::move(B_));

        mmul_command runner;
        // .run() blocks until the GPU finishes
        auto result_buffer = runner.run(program_,
                                        dim,
                                        self_->id(),
                                        arg1,
                                        arg2,
                                        out<int>(N * N),
                                        in(N));

        double duration = std::chrono::duration<double, std::milli>(
            clock::now() - start_).count();

        std::vector<int> output = caf::cuda::extract_vector<int>(result_buffer);
        return duration;
      },
    };
  }
};

void caf_main(actor_system& sys) {
  scoped_actor self{sys};
  std::vector<int> sizes = {1000, 2000, 4000, 8000, 16000};
  std::vector<double> results;

  // Warmup: prime CUDA context and cubin load before timed tests
  {
    int warmup_N = 64;
    auto w = self->spawn(caf::actor_from_state<MatMult>, warmup_N);
    self->mail(warmup_N).request(w, caf::infinite).receive(
      [](double) {}, [](const caf::error&) {});
  }
  std::cout << "--- warmup complete ---\n";

  for (int N : sizes) {
    auto worker = self->spawn(caf::actor_from_state<MatMult>, N);
    self->mail(N)
      .request(worker, caf::infinite)
      .receive(
        [&](double duration) {
          results.push_back(duration);
        },
        [&](const caf::error& err) {
          std::cerr << "Main: Error occurred for N=" << N
                    << ": " << to_string(err) << std::endl;
        }
      );
  }

  std::cout << "\nMatrix size : time (ms)\n";
  for (size_t i = 0; i < sizes.size(); ++i) {
    std::cout << sizes[i] << " : " << results[i] << " ms\n";
  }
}

CAF_MAIN(caf::cuda::manager)
