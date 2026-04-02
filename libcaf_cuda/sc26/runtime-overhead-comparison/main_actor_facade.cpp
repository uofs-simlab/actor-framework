#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <vector>
#include <iostream>
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
          using ms = std::chrono::duration<double, std::milli>;

          auto rp = self_->make_response_promise();

          auto t_total_start = clock::now();

          // -------------------------
          // create_in_arg A
          // -------------------------
          auto t_a_inarg_start = clock::now();
          auto inA = caf::cuda::create_in_arg(A_);
          auto t_a_inarg_end = clock::now();

          // -------------------------
          // create_in_arg B
          // -------------------------
          auto t_b_inarg_start = clock::now();
          auto inB = caf::cuda::create_in_arg(B_);
          auto t_b_inarg_end = clock::now();

          // -------------------------
          // create_out_arg C
          // -------------------------
          auto t_outC_start = clock::now();
          auto outC = caf::cuda::create_out_arg_with_size<int>(N * N);
          auto t_outC_end = clock::now();

          // -------------------------
          // create_in_arg N
          // -------------------------
          auto t_inN_start = clock::now();
          auto inN = caf::cuda::create_in_arg(N);
          auto t_inN_end = clock::now();

          // -------------------------
          // request (dispatch + GPU execution + D2H)
          // -------------------------
          auto t_request_start = clock::now();

          self_->mail(inA, inB, outC, inN)
              .request(gpuActor_, caf::infinite)
              .then(
                [this, rp, N,
                 t_total_start,
                 t_a_inarg_start, t_a_inarg_end,
                 t_b_inarg_start, t_b_inarg_end,
                 t_outC_start, t_outC_end,
                 t_inN_start, t_inN_end,
                 t_request_start](std::vector<output_buffer> result) mutable {
                  auto t_response_end = clock::now();

                  // -------------------------
                  // extract_vector
                  // -------------------------
                  auto t_extract_start = clock::now();
                  std::vector<int> output = caf::cuda::extract_vector<int>(std::move(result));
                  auto t_extract_end = clock::now();

                  auto t_total_end = clock::now();
                  double total_ms = ms(t_total_end - t_total_start).count();

                  std::cout << "\n===== ACTOR-FACADE BENCHMARK (N=" << N << ") =====\n";
                  std::cout << "create_in_arg A:                     "
                            << ms(t_a_inarg_end - t_a_inarg_start).count() << " ms\n";
                  std::cout << "create_in_arg B:                     "
                            << ms(t_b_inarg_end - t_b_inarg_start).count() << " ms\n";
                  std::cout << "create_out_arg C:                    "
                            << ms(t_outC_end - t_outC_start).count() << " ms\n";
                  std::cout << "create_in_arg N:                     "
                            << ms(t_inN_end - t_inN_start).count() << " ms\n";
                  std::cout << "request\u2192response (dispatch+GPU+D2H): "
                            << ms(t_response_end - t_request_start).count() << " ms\n";
                  std::cout << "extract_vector:                      "
                            << ms(t_extract_end - t_extract_start).count() << " ms\n";
                  std::cout << "TOTAL:                               "
                            << total_ms << " ms\n";
                  std::cout << "=============================================\n";

                  rp.deliver(total_ms);
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