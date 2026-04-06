#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include "../common/kernel_paths.hpp"
#include "vector"

class MatMult {
  caf::event_based_actor* self_;
  std::vector<int> A_ = {1, 2,
                         3, 4};
  std::vector<int> B_ = {5, 6,
                         7, 8};


  public:
    MatMult(caf::event_based_actor* self) : self_(self) {};

    caf::behavior make_behavior() {
      return {
        [this](int N) {

          int THREADS = 32;
          int BLOCKS = (N + THREADS - 1) / THREADS;

          // Launch dims: 2x2 threads is enough for a 2x2 product
          caf::cuda::nd_range dim(
              BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

          auto gpuActor = self_->system().cuda_manager().spawnFromCUBIN(
              actor_tests::paths::matmul_verbose_cubin, "matrixMul", dim,
              in<int>{}, in<int>{}, out<int>{}, in<int>{});

          self_->mail(
              caf::cuda::create_in_arg(A_), 
              caf::cuda::create_in_arg(B_), 
              caf::cuda::create_out_arg_with_size<int>(N * N), // Seems there is no need to create a host buffer, or maybe that is what this does?
              caf::cuda::create_in_arg(N))
              .send(gpuActor); 
        },

        [this](const std::vector<output_buffer>& result) {
          std::vector<int> output = caf::cuda::extract_vector<int>(result);
          self_->println("Received result from GPU actor {}, {}, {}, {} ", 
                             output[0], output[1], output[2], output[3]);
          self_->quit();
        }
      };
    }
};


void caf_main(caf::actor_system& sys) {
    caf::scoped_actor self{sys};
    self->println("Hello, CAF!");
    auto test_actor = self->spawn(caf::actor_from_state<MatMult>);
    self->mail(2).send(test_actor);
    self->await_all_other_actors_done();
}

CAF_MAIN()