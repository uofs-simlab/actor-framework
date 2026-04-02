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
        using ms = std::chrono::duration<double, std::milli>;

        int THREADS = 32;
        int BLOCKS = (N + THREADS - 1) / THREADS;

        caf::cuda::nd_range dim(BLOCKS,
                                BLOCKS,
                                1,
                                THREADS,
                                THREADS,
                                1);

        // Pre-allocate output buffer BEFORE the timed window so that all
        // pages are demand-faulted (backed by the OS) here, not inside
        // runner.run_into().  This mirrors the native benchmark which
        // pre-allocates h_c before t_total_start.
        std::vector<int> output(static_cast<size_t>(N) * N);

        auto t_total_start = clock::now();

        // -------------------------
        // create_in_arg A
        // -------------------------
        auto t_a_inarg_start = clock::now();
        auto arg1 = caf::cuda::create_in_arg(A_);
        auto t_a_inarg_end = clock::now();

        // -------------------------
        // create_in_arg B
        // -------------------------
        auto t_b_inarg_start = clock::now();
        auto arg2 = caf::cuda::create_in_arg(B_);
        auto t_b_inarg_end = clock::now();

        // -------------------------
        // runner.run_into (H2D A+B + kernel + D2H C into pre-allocated buffer)
        // -------------------------
        auto t_run_start = clock::now();

        mmul_command runner;
        runner.run_into(program_,
                        dim,
                        self_->id(),
                        output.data(),
                        static_cast<size_t>(N) * N,
                        arg1,
                        arg2,
                        caf::cuda::create_out_arg_with_size<int>(N * N),
                        caf::cuda::create_in_arg(N));

        auto t_run_end = clock::now();

        auto t_total_end = clock::now();
        double total_ms = ms(t_total_end - t_total_start).count();

        std::cout << "\n===== COMMAND-RUNNER BENCHMARK (N=" << N << ") =====\n";
        std::cout << "create_in_arg A:             "
                  << ms(t_a_inarg_end - t_a_inarg_start).count() << " ms\n";
        std::cout << "create_in_arg B:             "
                  << ms(t_b_inarg_end - t_b_inarg_start).count() << " ms\n";
        std::cout << "runner.run_into (H2D+kernel+D2H): "
                  << ms(t_run_end - t_run_start).count() << " ms\n";
        std::cout << "TOTAL:                       " << total_ms << " ms\n";
        std::cout << "=============================================\n";

        return total_ms;
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