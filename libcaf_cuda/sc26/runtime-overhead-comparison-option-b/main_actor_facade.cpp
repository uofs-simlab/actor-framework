// main_actor_facade.cpp
// Option B: Throughput mode — all REPS messages sent non-blocking before the
// actor processes any of them. The worker counts completed GPU results and
// quits after REPS are done. The outer scoped_actor calls wait_for() once —
// a single barrier matching native's single cuStreamSynchronize per series.
// This demonstrates the actor model's throughput ceiling.
#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

static const unsigned int RANDOM_SEED = 42;
static const int REPS = 10;   // repetitions per matrix size
using namespace std::chrono_literals;
using clock_t_ = std::chrono::steady_clock;


class MatMult {
  caf::event_based_actor* self_;
  std::vector<int> A_;
  std::vector<int> B_;
  caf::actor gpuActor_;
  int total_;       // total messages expected
  int completed_;   // GPU results received so far

public:
  MatMult(caf::event_based_actor* self, int N, int total)
      : self_(self), total_(total), completed_(0) {
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
        // Dispatch to GPU actor, discard result, count completions
        self_->mail(
            caf::cuda::create_in_arg(A_),
            caf::cuda::create_in_arg(B_),
            caf::cuda::create_out_arg_with_size<int>(N * N),
            caf::cuda::create_in_arg(N))
            .request(gpuActor_, caf::infinite)
            .then(
              [this](const std::vector<output_buffer>&) {
                ++completed_;
                if (completed_ >= total_) {
                  // All work done — quit so wait_for() in the caller unblocks
                  self_->quit();
                }
              },
              [](const caf::error& err) {
                std::cerr << "GPU error: " << to_string(err) << "\n";
              }
            );
      }
    };
  }
};


void caf_main(caf::actor_system& sys) {
  caf::scoped_actor self{sys};
  std::vector<int> sizes = {1000, 2000, 4000, 8000, 16000};

  // Warmup: send one message and wait for the worker to finish
  {
    auto w = self->spawn(caf::actor_from_state<MatMult>, 64, 1);
    caf::anon_mail(64).send(w);
    self->wait_for(w);
  }
  std::cout << "--- warmup complete ---\n";

  std::vector<std::pair<int,double>> results;

  for (int N : sizes) {
    // Worker knows how many results to expect before quitting
    auto worker = self->spawn(caf::actor_from_state<MatMult>, N, REPS);

    auto t_start = clock_t_::now();

    // Fire all REPS messages non-blocking — actor queues them in its mailbox
    // and processes back-to-back, matching native stream pipelining
    for (int r = 0; r < REPS; ++r) {
      caf::anon_mail(N).send(worker);
    }

    // Single barrier — blocks until worker has received all REPS GPU results
    self->wait_for(worker);

    auto t_end = clock_t_::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "\n===== ACTOR-FACADE THROUGHPUT (N=" << N
              << ", reps=" << REPS << ") =====\n";
    std::cout << "Total for " << REPS << " reps: " << total_ms << " ms\n";
    std::cout << "Per-rep average: " << total_ms / REPS << " ms\n";
    std::cout << "=============================================\n";

    results.emplace_back(N, total_ms);
  }

  std::cout << "\nMatrix size : total time (ms) for " << REPS << " reps : per-rep avg (ms)\n";
  for (auto &p : results) {
    std::cout << p.first << " : " << p.second
              << " ms : " << p.second / REPS << " ms\n";
  }
}

CAF_MAIN(caf::cuda::manager)
