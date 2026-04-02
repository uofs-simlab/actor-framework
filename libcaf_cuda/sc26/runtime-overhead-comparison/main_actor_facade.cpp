#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <vector>
#include <random>
#include <chrono>

static const unsigned int RANDOM_SEED = 42;
using namespace std::chrono_literals;


class MatMult {
  caf::event_based_actor* self_;
  std::vector<int> A_;
  std::vector<int> B_;
  caf::actor gpuActor_;
  using clock = std::chrono::steady_clock;
  clock::time_point start_;

  public:
    MatMult(caf::event_based_actor* self, int N) : self_(self) {
      // Initialize A and B with deterministic random data
      std::mt19937 rng(RANDOM_SEED);
      std::uniform_int_distribution<int> dist(1, 10);

      A_.resize(N * N);
      B_.resize(N * N);

      for (auto& v : A_) v = dist(rng);
      for (auto& v : B_) v = dist(rng);

      // Pre-load cubin and spawn GPU actor outside the timed window
      int THREADS = 32;
      int BLOCKS = (N + THREADS - 1) / THREADS;
      caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

      gpuActor_ = self_->system().cuda_manager().spawnFromCUBIN(
          "mmul.cubin", "matrixMul", dim,
          in<int>{}, in<int>{}, out<int>{}, in<int>{});
    };

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