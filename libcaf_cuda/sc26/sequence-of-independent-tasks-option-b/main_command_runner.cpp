// main_command_runner.cpp
// Option B: Throughput mode — all iterations sent non-blocking before the actor
// processes any of them. The worker counts completed dispatches and quits after
// 'iterations' are done. The outer scoped_actor calls wait_for() once — a
// single barrier matching native's single cuStreamSynchronize per series.
// Demonstrates the command-runner throughput ceiling.
#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>

static const unsigned int RANDOM_SEED = 42;
using namespace std::chrono_literals;
using clock_t_ = std::chrono::steady_clock;
using mmul_command = caf::cuda::command_runner<
    in<int>,  // matrix A
    in<int>,  // matrix B
    out<int>, // matrix C
    in<int>   // matrix size N
>;


class MatMult {
  caf::event_based_actor* self_;
  std::vector<int> A_;
  std::vector<int> B_;
  caf::cuda::program_ptr program_;
  int total_;       // total messages expected
  int completed_;   // dispatches completed so far
  double acc_create_inA_ms_ = 0;
  double acc_create_inB_ms_ = 0;
  double acc_run_ms_        = 0;
  double acc_extract_ms_    = 0;

public:
  MatMult(caf::event_based_actor* self, int N, int total)
      : self_(self), total_(total), completed_(0) {
    std::mt19937 rng(RANDOM_SEED);
    std::uniform_int_distribution<int> dist(1, 10);

    A_.resize(N * N);
    B_.resize(N * N);

    for (auto& v : A_) v = dist(rng);
    for (auto& v : B_) v = dist(rng);

    program_ = self_->system().cuda_manager().create_program_from_cubin(
        "mmul.cubin", "matrixMul");
  }

  caf::behavior make_behavior() {
    return {
      [this](int N) {
        using ms = std::chrono::duration<double, std::milli>;

        int THREADS = 32;
        int BLOCKS = (N + THREADS - 1) / THREADS;

        caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

        // -------------------------
        // create_in_arg A
        // -------------------------
        auto t_a_start = clock_t_::now();
        auto arg1 = caf::cuda::create_in_arg(A_);
        auto t_a_end = clock_t_::now();

        // -------------------------
        // create_in_arg B
        // -------------------------
        auto t_b_start = clock_t_::now();
        auto arg2 = caf::cuda::create_in_arg(B_);
        auto t_b_end = clock_t_::now();

        // -------------------------
        // runner.run (H2D A+B + kernel + D2H C)
        // -------------------------
        auto t_run_start = clock_t_::now();
        mmul_command runner;
        auto result_buffer = runner.run(program_,
                                        dim,
                                        self_->id(),
                                        arg1,
                                        arg2,
                                        caf::cuda::create_out_arg_with_size<int>(N * N),
                                        caf::cuda::create_in_arg(N));
        auto t_run_end = clock_t_::now();

        // -------------------------
        // extract_vector
        // -------------------------
        auto t_extract_start = clock_t_::now();
        (void)caf::cuda::extract_vector<int>(result_buffer);
        auto t_extract_end = clock_t_::now();

        acc_create_inA_ms_ += ms(t_a_end       - t_a_start).count();
        acc_create_inB_ms_ += ms(t_b_end       - t_b_start).count();
        acc_run_ms_        += ms(t_run_end     - t_run_start).count();
        acc_extract_ms_    += ms(t_extract_end - t_extract_start).count();

        ++completed_;
        if (completed_ >= total_) {
          double n = static_cast<double>(total_);
          std::cout << "\n[PHASE BREAKDOWN] Series iters=" << total_
                    << " (N=" << N << "):\n";
          std::cout << "  Avg create_in_arg A:             " << acc_create_inA_ms_ / n << " ms\n";
          std::cout << "  Avg create_in_arg B:             " << acc_create_inB_ms_ / n << " ms\n";
          std::cout << "  Avg runner.run (H2D+kernel+D2H): " << acc_run_ms_        / n << " ms\n";
          std::cout << "  Avg extract_vector:              " << acc_extract_ms_    / n << " ms\n";
          std::cout << "  Avg TOTAL per-iteration:         "
                    << (acc_create_inA_ms_ + acc_create_inB_ms_ + acc_run_ms_ + acc_extract_ms_) / n
                    << " ms\n";
          self_->quit();
        }
      }
    };
  }
};


void caf_main(caf::actor_system& sys) {
  caf::scoped_actor self{sys};
  const int N = 1000;
  std::vector<int> iteration_series = {1000, 2000, 3000, 4000, 5000,
                                       6000, 7000, 8000, 9000, 10000};

  // Warmup: send one fire-and-forget and wait for worker to finish
  {
    auto w = self->spawn(caf::actor_from_state<MatMult>, N, 1);
    caf::anon_mail(N).send(w);
    self->wait_for(w);
  }
  std::cout << "--- warmup complete ---\n";

  for (int iterations : iteration_series) {
    // Worker knows how many results to expect before quitting
    auto worker = self->spawn(caf::actor_from_state<MatMult>, N, iterations);

    auto start = clock_t_::now();

    // [OPTION B] Fire all iterations non-blocking
    for (int i = 0; i < iterations; ++i) {
      caf::anon_mail(N).send(worker);
    }

    // Single barrier
    self->wait_for(worker);

    auto end = clock_t_::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "[SERIES RESULT] Matrix " << N << "x" << N
              << ", iterations = " << iterations
              << ", total GPU time = " << total_ms << " ms\n";
  }
}

CAF_MAIN(caf::cuda::manager)
