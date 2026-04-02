// main_actor_facade.cpp
// Option A: Serialised latency — actor variant unchanged from the corrected
// sequence-of-independent-tasks test. request/receive is already blocking per-
// iteration, matching the native variant which now also synchronises per-
// iteration. This is the fairest apples-to-apples overhead comparison.
#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>

static const unsigned int RANDOM_SEED = 42;
using namespace std::chrono_literals;
using clock_t_ = std::chrono::steady_clock;


class MatMult {
  caf::event_based_actor* self_;
  std::vector<int> A_;
  std::vector<int> B_;
  caf::actor gpuActor_;
  using clock = std::chrono::steady_clock;
  clock::time_point start_;

public:
  MatMult(caf::event_based_actor* self, int N) : self_(self) {
    std::mt19937 rng(RANDOM_SEED);
    std::uniform_int_distribution<int> dist(1, 10);

    A_.resize(N * N);
    B_.resize(N * N);

    for (auto& v : A_) v = dist(rng);
    for (auto& v : B_) v = dist(rng);

    int THREADS = 32;
    int BLOCKS = (N + THREADS - 1) / THREADS;
    caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    gpuActor_ = self_->system().cuda_manager().spawnFromCUBIN(
        "mmul.cubin", "matrixMul", dim,
        in<int>{}, in<int>{}, out<int>{}, in<int>{});
  }

  caf::behavior make_behavior() {
    return {
      [this](int N) {
        start_ = clock::now();

        auto rp = self_->make_response_promise();

        self_->mail(
            caf::cuda::create_in_arg(A_),
            caf::cuda::create_in_arg(B_),
            caf::cuda::create_out_arg_with_size<int>(N * N),
            caf::cuda::create_in_arg(N))
            .request(gpuActor_, caf::infinite)
            .then(
              [this, rp](const std::vector<output_buffer>& result) mutable {
                std::vector<int> output = caf::cuda::extract_vector<int>(result);
                double duration = std::chrono::duration<double, std::milli>(
                    clock::now() - start_).count();
                rp.deliver(duration);
              },
              [rp](const caf::error& err) mutable {
                rp.deliver(err);
              }
            );
      }
    };
  }
};


void caf_main(caf::actor_system& sys) {
  caf::scoped_actor self{sys};
  int N = 1000;
  int iterations = 10000;
  int checkpoint = 1000;

  auto worker = self->spawn(caf::actor_from_state<MatMult>, N);

  // Warmup: prime CUDA context before timed measurements
  self->mail(N).request(worker, caf::infinite).receive(
    [](double) {},
    [](const caf::error& err) {
      std::cerr << "Warmup error: " << to_string(err) << std::endl;
    });
  std::cout << "--- warmup complete ---\n";

  auto loop_start = clock_t_::now();

  for (int i = 0; i < iterations; i++) {
    self->mail(N)
      .request(worker, caf::infinite)
      .receive(
        [](double /*per_iter_ms*/) {},
        [&](const caf::error& err) {
          std::cerr << "Main: Error occurred for N=" << N
                    << ": " << to_string(err) << std::endl;
        }
      );

    if ((i + 1) % checkpoint == 0) {
      auto now = clock_t_::now();
      double elapsed_ms = std::chrono::duration<double, std::milli>(now - loop_start).count();
      std::cout << "[SERIES RESULT] Matrix " << N << "x" << N
                << ", iterations = " << (i + 1)
                << ", total GPU time = " << elapsed_ms << " ms\n";
    }
  }
}

CAF_MAIN(caf::cuda::manager)
