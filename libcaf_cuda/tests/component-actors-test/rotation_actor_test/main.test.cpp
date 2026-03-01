#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/component-actors/rotation_actor/rotation_actor.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace caf;
using namespace std::chrono_literals;

template <typename T>
void rotate_points_cpu(const std::vector<T>& rotation_matrix,
                       const std::vector<T>& points,
                       std::vector<T>& result) {
    // points: [x0, x1, ..., y0, y1, ...] row-major 2xM
    int M = points.size() / 2;
    for (int i = 0; i < M; ++i) {
        T x = points[i];
        T y = points[M + i];

        // Apply rotation: [x', y'] = R * [x, y]
        result[i] = rotation_matrix[0] * x + rotation_matrix[1] * y;       // x'
        result[M + i] = rotation_matrix[2] * x + rotation_matrix[3] * y;   // y'
    }
}



caf::cuda::command_runner<> runner;

struct supervisor_state {};

// Supervisor actor behavior
caf::behavior rotation_supervisor(caf::stateful_actor<supervisor_state> * self,
                                    caf::cuda::program_ptr program) {
   
            caf::actor worker = self ->spawn(caf::cuda::rotation_actor_fun<float>, program);
       
	return {
        [=](std::vector<float> rotation_matrix,
		std::vector<float> points) mutable {

            // Spawn the vector add worker actor
            //caf::actor worker = self ->spawn(caf::cuda::vector_add_actor_fun<int>, program);

            // Transfer memory to device
		caf::cuda::mem_ptr<float> rotation_matrix_memory = runner.transfer_memory(0,1,caf::cuda::create_in_arg(rotation_matrix));
		caf::cuda::mem_ptr<float>  points_memory = runner.transfer_memory(0,1,caf::cuda::create_in_arg(points));

	    int num_points = points.size() / 2;
            // Send to worker actor and request result
            self->mail(rotation_matrix_memory,
                                 points_memory,
                                 num_points,
                                 0,
                                 1) // device=0, stream=1
                .request(worker, std::chrono::seconds(10))
                .then([=](caf::cuda::mem_ptr<float> rotated_points) {
                    std::vector<float> result(points.size());
                    std::vector<float> vecC = rotated_points->copy_to_host();

                    rotate_points_cpu(rotation_matrix, points, result);

                    if (result == vecC)
                        std::cout << "rotation result matches!\n";
                    else
                        std::cout << "rotation result mismatch!\n";

                    // Quit the supervisor
                    self->quit();
                });
        }
    };
}

void run_vector_add_test(actor_system& sys, int vec_size) {
    caf::cuda::manager::init(sys);

    // Create program pointer for the vector add kernel
    auto program = caf::cuda::manager::get()
                       .create_program_from_cubin("../mmul.cubin", "mmul_non_square");


  
    // Generate test data

    std::vector<float> rotation_matrix = {
	    0.0f, -1.0f,  // first row
	    1.0f,  0.0f   // second row
    };


    std::vector<float> points = {
    1.0f, 2.0f, 4.0f, 8.0f,   // x-coordinates
    1.0f, 2.0f, 4.0f, 8.0f    // y-coordinates
};

    
    // Spawn the supervisor actor
    caf::actor supervisor = sys.spawn(rotation_supervisor,program);

    // Trigger the supervisor
    anon_mail(rotation_matrix,points).send(supervisor);

    // Wait for all actors to finish
    sys.await_all_actors_done();
}

void caf_main(actor_system& sys) {
    run_vector_add_test(sys, 1024);
}

CAF_MAIN()
