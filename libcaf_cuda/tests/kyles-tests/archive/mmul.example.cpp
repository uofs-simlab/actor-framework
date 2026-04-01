#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include "vector"


using namespace caf;
using namespace std::chrono_literals;

//verification of matrix multiplication on the gpu
void serial_matrix_multiply(const std::vector<int>& a,
                            const std::vector<int>& b,
                            std::vector<int>& c,
                            int N) {  
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int sum = 0;
      for (int k = 0; k < N; ++k) {
        sum += a[i * N + k] * b[k * N + j];
      }
      c[i * N + j] = sum;
    }
  }
}

caf::behavior testee(caf::event_based_actor* self) {
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
        auto arg3 = caf::cuda::create_out_arg_with_size<int>(elems);
        auto arg4 = caf::cuda::create_in_arg(N);

        self->mail(arg1,
                arg2,
                arg3,
                arg4)
            .request(gpuActor, std::chrono::seconds(5))
            .then([=](const std::vector<output_buffer>& result) {
                std::vector<int> output = caf::cuda::extract_vector<int>(result);
                self->println("Received result from GPU actor {}, {}, {}, {} ", output[0], output[1], output[2], output[3]);
                self->quit();
            });
  return {
    // [=](int x) {

    // }
  };
}


// void test_mmul_from_cubin(caf::actor_system& sys, int N) {
//   std::cout << "[TEST] Starting test_mmul_from_cubin\n";

//   caf::cuda::manager& mgr = caf::cuda::manager::get();

//   int THREADS = 32;
//   int BLOCKS = (N + THREADS - 1) / THREADS;

//   caf::cuda::nd_range dim(
// 		  BLOCKS, //grid X dimension
// 		  BLOCKS, //grid Y dimension
// 		  1, //grid Z dimension
// 		  THREADS, // block X dimension 
// 		  THREADS, // block Y dimension
// 		  1 // block Z dimension
// 		  );


//   // Spawn actor from precompiled cubin file
//   auto gpuActor = mgr.spawnFromCUBIN(
// 		  "mmul.cubin", //kernel file location
// 		  "matrixMul", //kernel name 
// 		  dim, //kernel dimensions             
// 		  in<int>{}, in<int>{}, out<int>{}, in<int>{} //kernel arg tags 
// 							      //in order they appear in kernel
// 		  );




//   //generate random matrices
//   std::vector<int> h_a(N * N);
//   std::vector<int> h_b(N * N);
//   std::vector<int> h_c(N * N, 0);
//   std::vector<int> h_ref(N * N, 0);
//   std::vector<int> h_n(1, N);

//   h_a = {1, 2,
//          3, 4};
//   h_b = {5, 6,
//          7, 8};



//   //tag the arguments 
//   auto arg1 = caf::cuda::create_in_arg(h_a); //matrix A readonly buffer
//   auto arg2 = caf::cuda::create_in_arg(h_b); //matrix B readonly buffer
//   auto arg3 = caf::cuda::create_out_arg_with_size<int>(N*N); //matrix size Writeonly buffer
//   auto arg4 = caf::cuda::create_in_arg(N); // int size, readonly scalar

//   serial_matrix_multiply(h_a, h_b, h_ref, N);
//   sys.println("Reference result {}, {}, {}, {} ", h_ref[0], h_ref[1], h_ref[2], h_ref[3]);

//   sys.spawn([=](caf::event_based_actor* self_actor) {
//     auto start = std::chrono::high_resolution_clock::now();

//     //when mailing the gpu actor, the message is in the form of the kernel arguments 
//     //and must be in the order they appear in the kernel parameters 
//     //it will deliever a response promise with the results of that kernel launch
//     self_actor->mail(arg1, arg2, arg3, arg4)
//       .request(gpuActor, std::chrono::seconds(10))
//       .then([=](const std::vector<output_buffer>& outputs) {
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsed = end - start;

	
//         std::vector<int> result = caf::cuda::extract_vector<int>(outputs); //collect the result buffer from output

//         self_actor->println("Received result from GPU actor {}, {}, {}, {} ", result[0], result[1], result[2], result[3]);


//         // Compare result with reference
//         bool match = (result == h_ref);
//         std::cout << "[INFO] Kernel round-trip time: " << elapsed.count() << " seconds\n";
//         std::cout << (match ? "[PASS] GPU result matches reference\n" : "[FAIL] Mismatch in GPU result\n");

//         self_actor->send_exit(gpuActor, caf::exit_reason::user_shutdown);
//         self_actor->quit();
        
//       });
//   });

//   sys.await_all_actors_done();
// }




void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);  //be sure to initialize the manager
				  //as certain things need to be startup 
  //  test_mmul_from_cubin(sys,2);
  scoped_actor self{sys};

  auto test = self->spawn(testee);
  self->mail(42).send(test);
  self->await_all_other_actors_done();

   //test_mmul_from_cubin(sys,50);
   //test_mmul_from_cubin(sys,1024);

}




CAF_MAIN()
