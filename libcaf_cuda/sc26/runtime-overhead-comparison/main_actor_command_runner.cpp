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
  std::vector<int> C_;
  using clock = std::chrono::steady_clock;
  using ms = std::chrono::duration<double, std::milli>;
  clock::time_point start_, end_;

public:
  MatMult(event_based_actor* self, int N) : self_(self) {
    std::mt19937 rng(RANDOM_SEED);
    std::uniform_int_distribution<int> dist(1, 10);

    A_.resize(N * N);
    B_.resize(N * N);

    for (auto& v : A_) v = dist(rng);
    for (auto& v : B_) v = dist(rng);

    C_.resize(N * N);
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
      },
    };
  }
};

void caf_main(actor_system& sys) {
  scoped_actor self{sys};
  std::vector<int> sizes = {1000, 2000, 4000, 8000, 16000};
  std::vector<double> results;
  for (int N : sizes) {
    auto worker = self->spawn(caf::actor_from_state<MatMult>, N);
    self->mail(N)
      .request(worker, caf::infinite)
      .receive(
        [&](double duration) {
          results.push_back(duration);
          self->println("Duration = {}", duration);
        },
        [&](const caf::error& err) {
          std::cerr << "Main: Error occurred for N=" << N 
                    << ": " << to_string(err) << std::endl;
        }
      );
  }
  // Print summary of results
  std::cout << "\nMatrix size : time (ms)\n";
  for (size_t i = 0; i < sizes.size(); ++i) {
    std::cout << sizes[i] << " : " << results[i] << " ms"
              << std::endl;
  }
}

CAF_MAIN(caf::cuda::manager)