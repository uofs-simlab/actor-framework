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
  using clock = std::chrono::steady_clock;
  clock::time_point start_;
  int iter_count_ = 0;
  static constexpr int checkpoint_ = 1000;
  double acc_create_inA_ms_ = 0;
  double acc_create_inB_ms_ = 0;
  double acc_run_ms_        = 0;
  double acc_extract_ms_    = 0;
  double acc_total_ms_      = 0;


public:
  MatMult(caf::event_based_actor* self, int N) : self_(self) {
    // Initialize persistent host buffers (matches native CUDA's single allocation)
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

  caf::behavior make_behavior() {
    return {
      [this](int N) {
        using ms = std::chrono::duration<double, std::milli>;

        int THREADS = 32;
        int BLOCKS = (N + THREADS - 1) / THREADS;

        caf::cuda::nd_range dim(BLOCKS,
                                BLOCKS,
                                1,
                                THREADS,
                                THREADS,
                                1);

        auto t_total_start = clock::now();

        // -------------------------
        // create_in_arg A
        // -------------------------
        auto t_a_start = clock::now();
        auto arg1 = caf::cuda::create_in_arg(A_);
        auto t_a_end = clock::now();

        // -------------------------
        // create_in_arg B
        // -------------------------
        auto t_b_start = clock::now();
        auto arg2 = caf::cuda::create_in_arg(B_);
        auto t_b_end = clock::now();

        // -------------------------
        // runner.run (H2D A+B + kernel + D2H C)
        // -------------------------
        auto t_run_start = clock::now();
        mmul_command runner;
        auto result_buffer = runner.run(program_,
                                        dim,
                                        self_->id(),
                                        arg1,
                                        arg2,
                                        caf::cuda::create_out_arg_with_size<int>(N * N),
                                        caf::cuda::create_in_arg(N));
        auto t_run_end = clock::now();

        // -------------------------
        // extract_vector
        // -------------------------
        auto t_extract_start = clock::now();
        std::vector<int> output = caf::cuda::extract_vector<int>(result_buffer);
        auto t_extract_end = clock::now();

        auto t_total_end = clock::now();
        double total_ms = ms(t_total_end - t_total_start).count();

        acc_create_inA_ms_ += ms(t_a_end       - t_a_start).count();
        acc_create_inB_ms_ += ms(t_b_end       - t_b_start).count();
        acc_run_ms_        += ms(t_run_end     - t_run_start).count();
        acc_extract_ms_    += ms(t_extract_end - t_extract_start).count();
        acc_total_ms_      += total_ms;

        ++iter_count_;
        if (iter_count_ % checkpoint_ == 0) {
          double n = static_cast<double>(checkpoint_);
          std::cout << "\n[PHASE BREAKDOWN] After " << iter_count_
                    << " iterations (N=" << N << "), last " << checkpoint_ << " iters:\n";
          std::cout << "  Avg create_in_arg A:             " << acc_create_inA_ms_ / n << " ms\n";
          std::cout << "  Avg create_in_arg B:             " << acc_create_inB_ms_ / n << " ms\n";
          std::cout << "  Avg runner.run (H2D+kernel+D2H): " << acc_run_ms_        / n << " ms\n";
          std::cout << "  Avg extract_vector:              " << acc_extract_ms_    / n << " ms\n";
          std::cout << "  Avg TOTAL per-iteration:         " << acc_total_ms_      / n << " ms\n";
          // Reset accumulators for next checkpoint window
          acc_create_inA_ms_ = acc_create_inB_ms_ = acc_run_ms_
                             = acc_extract_ms_    = acc_total_ms_ = 0;
        }

        return total_ms;
      }
    };
  }
};


void caf_main(caf::actor_system& sys) {
  caf::scoped_actor self{sys};
  int N = 1000;
  int iterations = 10000;
  int checkpoint = 1000;

  // Spawn one persistent actor reused across all iterations
  // (persistent host buffers match native CUDA's single allocation)
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