#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include "vector"

using namespace caf;

caf::behavior testee(caf::event_based_actor* self) {
  return {
    [=](int x) {
        caf::cuda::manager& mgr = caf::cuda::manager::get();

        const int N = 2;
        int THREADS = 32;
        int BLOCKS = (N + THREADS - 1) / THREADS;

        const int elems = N * N;

        // 2x2 input matrices (row-major)
        std::vector<int> h_a = {1, 2,
                                3, 4};
        std::vector<int> h_b = {5, 6,
                                7, 8};

        // Output buffer (will be filled by GPU)
        std::vector<int> h_c(elems, 0);

        // Launch dims: 2x2 threads is enough for a 2x2 product
        caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, 
                                THREADS, THREADS, 1);

        auto gpuActor = mgr.spawnFromCUBIN("matmul.cubin",
                                            "matrixMul",
                                            dim,
                                            in<int>{}, in<int>{}, out<int>{}, in<int>{});

        self->println("{}, {}, {}, {}", h_a[0], h_a[1], h_a[2], h_a[3]);
        auto arg1 = caf::cuda::create_in_arg(h_a);
        auto arg2 = caf::cuda::create_in_arg(h_b);
        auto arg

        self->mail(arg1,
                caf::cuda::create_in_arg(h_b),
                caf::cuda::create_out_arg_with_size<int>(elems),
                caf::cuda::create_in_arg(N))
            .request(gpuActor, std::chrono::seconds(5))
            .then([=](const std::vector<output_buffer>& result) {
                std::vector<int> output = caf::cuda::extract_vector<int>(result);
                self->println("Received result from GPU actor {}, {}, {}, {} ", 
                             output[0], output[1], output[2], output[3]);
                self->quit();
            });
    }
  };
}


void caf_main(caf::actor_system& sys) {
    scoped_actor self{sys};
    caf::cuda::manager::init(sys);
    self->println("Hello, CAF!");
    auto test_actor = sys.spawn(testee);
    self->mail(42).send(test_actor);
    self->await_all_other_actors_done();
}

CAF_MAIN()