// Stateful Actor Example: Matrix Multiplication
// This example demonstrates how to use CAF's CUDA integration to perform matrix multiplication on the GPU.
// Answer should be 19, 22, 43, 50

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include "vector"
#include <random>

static const unsigned int RANDOM_SEED = 42;
using namespace std::chrono_literals;


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
      self->println("MatMult actor created with N={}", N);
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

          auto rp = self_->make_response_promise();

          int THREADS = 32;
          int BLOCKS = (N + THREADS - 1) / THREADS;

          caf::cuda::nd_range dim(
              BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

          auto gpuActor = self_->system().cuda_manager().spawnFromCUBIN(
              "mmul.cubin", "matrixMul", dim,
              in<int>{}, in<int>{}, out<int>{}, in<int>{});

          self_->mail(
              caf::cuda::create_in_arg(A_), 
              caf::cuda::create_in_arg(B_), 
              caf::cuda::create_out_arg_with_size<int>(N * N), // Seems there is no need to create a host buffer, or maybe that is what this does?
              caf::cuda::create_in_arg(N))
              .request(gpuActor, caf::infinite)
              .then(
                [this, rp](const std::vector<output_buffer>& result) mutable {
                  std::vector<int> output = caf::cuda::extract_vector<int>(result);
                  end_ = clock::now();
                  double duration = std::chrono::duration_cast<
                      std::chrono::milliseconds>(
                        end_ - start_).count();
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
  std::vector<int> sizes = {1000, 2000, 4000, 8000, 16000};
  std::vector<double> results;
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
  // Print summary of results
  std::cout << "\nMatrix size : time (ms)\n";
  for (size_t i = 0; i < sizes.size(); ++i) {
    std::cout << sizes[i] << " : " << results[i] << " ms"
              << std::endl;
  }

}

CAF_MAIN(caf::cuda::manager)