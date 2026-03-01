#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/component-actors/vector_add_actor/vector_add_actor.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace caf;
using namespace std::chrono_literals;

// Simple serial vector addition for verification
template <typename T>
void serial_vector_add(const std::vector<T>& a,
                       const std::vector<T>& b,
                       std::vector<T>& c) {
    for (size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}


caf::cuda::command_runner<> runner;

struct supervisor_state {};

// Supervisor actor behavior
caf::behavior vector_add_supervisor(caf::stateful_actor<supervisor_state> * self,
                                    const std::vector<int>& vecA,
                                    const std::vector<int>& vecB,
                                    int vec_size,
                                    caf::cuda::program_ptr program) {
   
            caf::actor worker = self ->spawn(caf::cuda::vector_add_actor_fun<int>, program);
       
	return {
        [&,self,worker,vecA,vecB,vec_size](const unit_t&) mutable {

            // Spawn the vector add worker actor
            //caf::actor worker = self ->spawn(caf::cuda::vector_add_actor_fun<int>, program);

            // Transfer memory to device
            auto dA = runner.transfer_memory(0,1,caf::cuda::create_in_arg(vecA));
            auto dB = runner.transfer_memory(0,1,caf::cuda::create_in_arg(vecB));

            // Send to worker actor and request result
            self->mail(dA, dB, vec_size, 0, 1) // device=0, stream=1
                .request(worker, std::chrono::seconds(10))
                .then([&](caf::cuda::mem_ptr<int> dC) {
                    std::vector<int> result(vec_size);
                    std::vector<int> vecC = dC->copy_to_host();

                    serial_vector_add(vecA, vecB, result);

                    if (result == vecC)
                        std::cout << "Vector addition result matches!\n";
                    else
                        std::cout << "Vector addition result mismatch!\n";

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
                       .create_program_from_cubin("../vector_add.cubin", "vectorAdd");

    // Generate test data
    std::vector<int> vecA(vec_size, 2);
    std::vector<int> vecB(vec_size, 3);

    // Spawn the supervisor actor
    caf::actor supervisor = sys.spawn(vector_add_supervisor,
                                       vecA, vecB, vec_size, program);

    // Trigger the supervisor
    anon_mail(unit).send(supervisor);

    // Wait for all actors to finish
    sys.await_all_actors_done();
}

void caf_main(actor_system& sys) {
    run_vector_add_test(sys, 1024);
}

CAF_MAIN()
