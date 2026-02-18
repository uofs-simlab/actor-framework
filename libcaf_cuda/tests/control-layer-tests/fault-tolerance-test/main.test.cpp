#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/cuda/control-layer/all-control-layer.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <random>
#include <unistd.h>
#include "caf/actor_registry.hpp"
#include <chrono>
#include <iostream>
//#include <caf/atoms.hpp>




using namespace caf;
using namespace std::chrono_literals;


struct exit_actor_state {
	int completed = 0;
};


// --- command runner types (put near top of file) -------------------------
using initCommand =
  caf::cuda::command_runner<caf::cuda::mem_ptr<float>, in<int>, in<unsigned long long>>;

using divCommand  =
  caf::cuda::command_runner<caf::cuda::mem_ptr<float>, caf::cuda::mem_ptr<float>, caf::cuda::mem_ptr<float>, in<int>>;

using sumCommand  =
  caf::cuda::command_runner<caf::cuda::mem_ptr<float>, caf::cuda::mem_ptr<float>, in<int>>;

// single instances (can be file-global)
static initCommand init_cmd;
static divCommand  div_cmd;
static sumCommand  sum_cmd;

// --- pipeline actor state (device buffers persist here) ------------------
struct pipeline_actor_state {
    int id = rand();

    // device-side buffers that must persist across stages:
    caf::cuda::mem_ptr<float> d_denoms;
    caf::cuda::mem_ptr<float> d_results;
    caf::cuda::mem_ptr<float> d_sum;
};

// --- corrected pipeline_actor -------------------------------------------
behavior pipeline_actor(caf::stateful_actor<pipeline_actor_state>* self,
                        actor supervisor,
                        caf::cuda::program_ptr p1,
                        caf::cuda::program_ptr p2,
                        caf::cuda::program_ptr p3,
                        int n)
{
    // host-side scratch (only used for post-stage2 NaN/Inf detection)
    std::vector<float> h_results;

    // scheduler from manager
    caf::actor scheduler = caf::cuda::manager::get().get_scheduler_actor();

    // nd_range used for all stages (adapt to your kernels as needed)
    caf::cuda::nd_range range{
        {(n + 255) / 256, 1, 1},
        {256, 1, 1}
    };

    // helper to create and send a launch token
    auto launch = [&](caf::cuda::program_ptr prog, const std::string& stage) {
        auto tok = make_launch_token(
            prog,
            range,
            /*memory_usage=*/static_cast<int>(sizeof(float) * n),
            stage,
            self,
            self->state().id  // dependency/demo id
        );
        anon_mail(tok).send(scheduler);
    };

    // fire all three tokens (scheduler will reply with response_token on grants)
    launch(p1, "stage1");
    launch(p2, "stage2");
    launch(p3, "stage3");

    return {

        // handle response tokens by name — opaque to reclaim payload
        [=](caf::cuda::response_token_ptr res_token) mutable {

            const auto& stage = res_token->name();

            // --------------------- Stage 1: init_denominators ---------------------
            if (stage == "stage1") {
                // allocate device buffer for denominators (persist in state)

     	           std::cout << "Starting stage 1\n";		    
		    unsigned long long seed = static_cast<unsigned long long>(
    std::chrono::high_resolution_clock::now().time_since_epoch().count()
);



		out<float> buffer = caf::cuda::create_out_arg_with_size<float>(n);
                self->state().d_denoms = init_cmd.transfer_memory(res_token,buffer);

                // run kernel on the stream/device from res_token
                // kernel signature: (float* denominators, int n, unsigned long long seed)
                init_cmd.run_async(
                    p1,
                    range,
                    res_token,                            // uses token's stream/device
                    self->state().d_denoms,               // device buffer
                    caf::cuda::create_in_arg(n),          // n
                    caf::cuda::create_in_arg(seed)     // seed
                );

		std::cout << "Finished stage 1\n";
                // stage1 intentionally no checks — data may contain zeros
                return;
            }

            // --------------------- Stage 2: perform_division ---------------------
            if (stage == "stage2") {
                // allocate device buffer for results (persist in state)
		    std::vector<float> buffer1(n); 

     	           std::cout << "Starting stage 2\n";		    
		self->state().d_results = div_cmd.transfer_memory(res_token,out<float>{buffer1});

                // create a host numerators vector (all ones)
                std::vector<float> h_nums(n, 1.0f);

                // transfer numerators to device on the token's stream/device
                // transfer_memory returns a caf::cuda::mem_ptr<float>
                auto d_nums = div_cmd.transfer_memory(res_token, in_out<float>{h_nums});

                // run division kernel on the token's stream/device:
                // kernel signature: (float* numerators, float* denominators, float* results, int n)
                div_cmd.run(
                    p2,
                    range,
                    res_token,
                    d_nums,
                    self->state().d_denoms,
                    self->state().d_results,
                    caf::cuda::create_in_arg(n)
                );

                // extract the device results back to host for verification.
                // extract_vector will synchronize as needed.
                h_results = self->state().d_results -> copy_to_host();

                // check for NaN/Inf AFTER the kernel finished
                bool fault = false;
                for (float v : h_results) {
                    if (!std::isfinite(v)) {
                        fault = true;
                        break;
                    }
                }

                if (fault) {
                    // inform supervisor and exit;
                    anon_mail(std::string("crash")).send(supervisor);
                    self->quit();
                    return;
                }

                // stage2 passed — keep d_results in state for stage3
                return;
            }

            // --------------------- Stage 3: sum_results --------------------------
            if (stage == "stage3") {
                // allocate device scalar for sum result
                
     	           std::cout << "Starting stage 3\n";		    
		    std::vector<float> buffer1(1); 

		self->state().d_sum = div_cmd.transfer_memory(res_token,out<float>{buffer1});
		    
                // run reduction on the token's stream/device:
                // kernel signature: (float* results, float* final_sum, int n)
                sum_cmd.run(
                    p3,
                    range,
                    res_token,
                    self->state().d_results,
                    self->state().d_sum,
                    caf::cuda::create_in_arg(n)
                );

                // extract final scalar
		
		std::vector<float> buf = self->state().d_sum -> copy_to_host();
                float final_sum = buf[0];
                std::cout << "[pipeline] completed, sum = " << final_sum << "\n";

                // successful completion -> tell supervisor to tear everything down
                anon_mail(std::string("done")).send(supervisor);

                // quit the pipeline actor
                self->quit();
                return;
            }

            // unknown stage: ignore or log
            std::cerr << "[pipeline] received unknown response token: " << stage << "\n";
        }
    };
}



void supervisor_handle_msg(event_based_actor* self,
                           caf::cuda::program_ptr p1,
                           caf::cuda::program_ptr p2,
                           caf::cuda::program_ptr p3,
                           int n,
                           const std::string& msg) {
    if (msg == "crash") {
        std::cout << "[supervisor] Pipeline crashed — restarting\n";
        self->system().spawn(pipeline_actor, self, p1, p2, p3, n);
    } else if (msg == "done") {
        std::cout << "[supervisor] Pipeline completed — shutting down\n";
        caf::cuda::manager::shutdown();
        self->quit();
    } else {
        std::cerr << "[supervisor] Unknown message: " << msg << "\n";
    }
}




behavior supervisor_actor(event_based_actor* self,
                          caf::cuda::program_ptr p1,
                          caf::cuda::program_ptr p2,
                          caf::cuda::program_ptr p3,
                          int n) {
    // Spawn first pipeline safely
    self->system().spawn(pipeline_actor, self, p1, p2, p3, n);

    // Behavior: just route string messages to the helper
    return {
        [=](const std::string& msg) {
            supervisor_handle_msg(self, p1, p2, p3, n, msg);
        }
    };
}




void caf_main(caf::actor_system& sys) {



        caf::cuda::manager_config man_config(true); //turns the scheduler on
        caf::cuda::manager::init(sys,man_config);

	//change the scheduler to core_usage
	anon_mail(
        caf::cuda::make_behavior_token("single_usage")
        ).send(caf::cuda::manager::get().get_scheduler_actor());


	caf::cuda::program_ptr p1 = caf::cuda::manager::get().create_program_from_cubin("../fault.cubin","init_denominators");
	caf::cuda::program_ptr p2 = caf::cuda::manager::get().create_program_from_cubin("../fault.cubin","perform_division");
	caf::cuda::program_ptr p3 = caf::cuda::manager::get().create_program_from_cubin("../fault.cubin","sum_results");

	sys.spawn(supervisor_actor,p1,p2,p3,1);
	sys.await_all_actors_done();


}




CAF_MAIN()








