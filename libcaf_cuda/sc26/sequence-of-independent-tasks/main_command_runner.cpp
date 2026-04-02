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
  std::vector<int> C_;
  using clock = std::chrono::steady_clock;
  using ms = std::chrono::duration<double, std::milli>;
  clock::time_point start_, end_;


public:
  MatMult(caf::event_based_actor* self, int N) : self_(self) {
    // Initialize A and B with deterministic random data
    std::mt19937 rng(RANDOM_SEED);
    std::uniform_int_distribution<int> dist(1, 10);

    A_.resize(N * N);
    B_.resize(N * N);

    for (auto& v : A_) v = dist(rng);
    for (auto& v : B_) v = dist(rng);

    C_.resize(N * N);
  };

  caf::behavior make_behavior() {
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


        auto& mgr = self_->system().cuda_manager();
        auto program = mgr.create_program_from_cubin(
            "mmul.cubin", "matrixMul");


        mmul_command runner;

        // .run() is expected to block the actor's execution until the GPU finishes
        auto result_buffer = runner.run(program, 
                                        dim, 
                                        self_->id(), 
                                        caf::cuda::create_in_arg(A_),
                                        caf::cuda::create_in_arg(B_),
                                        caf::cuda::create_out_arg_with_size<int>(N * N),
                                        caf::cuda::create_in_arg(N));

        end_ = clock::now();
        double duration = std::chrono::duration_cast<
            std::chrono::milliseconds>(end_ - start_).count();

        std::vector<int> output = caf::cuda::extract_vector<int>(result_buffer);
        return duration;
      }
    };
  }
};


void caf_main(caf::actor_system& sys) {
  caf::scoped_actor self{sys};
  int N = 1000;
  int iterations = 10000;
  int checkpoint = 1000;

  auto loop_start = clock_t_::now();

  for (int i = 0; i < iterations; i++) {
    auto worker = self->spawn(caf::actor_from_state<MatMult>, N);
    self->mail(N)
      .request(worker, caf::infinite)
      .receive(
        [&](double /*per_iter_ms*/) {},
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