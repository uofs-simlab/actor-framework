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


// outside any function
std::vector<int> big_buffer;


using namespace caf;
using namespace std::chrono_literals;




struct memory_hog_actor_state {
	std::size_t bytes;
};

//commands classes used to launch kernels 
using mmulCommand = caf::cuda::command_runner<in<int>,in<int>,out<int>,in<int>>;
using matrixGenCommand = caf::cuda::command_runner<out<int>,in<int>,in<int>,in<int>>;

using mmulAsyncCommand = caf::cuda::command_runner<caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int>,out<int>,in<int>>;

mmulCommand mmul;
matrixGenCommand randomMatrix;
mmulAsyncCommand mmulAsync;



caf::behavior memory_hog_actor_fun(caf::stateful_actor<memory_hog_actor_state>* self,
                                   caf::actor exit_actor,
                                   std::size_t bytes) { 
    self->state().bytes = bytes; 
    caf::cuda::manager& mgr = caf::cuda::manager::get();
    caf::actor memory_actor = mgr.get_memory_actor();

    // send a memory request token
    caf::cuda::memory_request_token request(bytes, 0, self);
    self->mail(request).send(memory_actor);

    std::cout << "Booting\n";
    return {
        [=](caf::cuda::mem_token msg) {
            caf::cuda::command_runner<> mem_transfer_command;

            try {
                std::cout << "Grabbing onto memory\n";

                // use global big_buffer directly
                caf::cuda::mem_ptr<int> temp = 
                    mem_transfer_command.transfer_memory(0, 1, in_out<int>{big_buffer});

                std::cout << "Memory transfer successful!\n";
            }
            catch (const std::runtime_error& e) {
                std::cerr << "[ERROR] Runtime exception during memory transfer: " << e.what() << "\n";
            }
            catch (const std::bad_alloc& e) {
                std::cerr << "[ERROR] Allocation failed during memory transfer: " << e.what() << "\n";
            }
            catch (const std::exception& e) {
                std::cerr << "[ERROR] Exception during memory transfer: " << e.what() << "\n";
            }
            catch (...) {
                std::cerr << "[ERROR] Unknown exception during memory transfer\n";
            }

            // hold onto memory for a few seconds
            std::cout << "Memory hog holding onto memory for 5 seconds\n";
            std::this_thread::sleep_for(std::chrono::seconds(5));
            std::cout << "Memory hog releasing memory\n";

            self->mail(1).send(exit_actor);
            self->quit();
        },
    };
}

void run_memory_hog_test(caf::actor_system& sys, std::size_t bytes, int num_actors) {
  if (num_actors < 1) {
    std::cerr << "[ERROR] Number of actors must be >= 1\n";
    return;
  }

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  caf::actor exit_actor = mgr.spawn_exit_actor(num_actors);
  // Spawn num_actors actors running the mmul behavior
  std::vector<caf::actor> actors;
  big_buffer.resize(bytes / sizeof(int));
  actors.reserve(num_actors);
  for (int i = 0; i < num_actors; i++){
    actors.push_back(sys.spawn(memory_hog_actor_fun,exit_actor,bytes));
  }

   sys.await_all_actors_done();
}



void caf_main(caf::actor_system& sys) {
  
	

	caf::cuda::manager_config man_config(false,true); //turns the memory actor on
	caf::cuda::manager::init(sys,man_config);
       	run_memory_hog_test(sys,6221225472,8);
}




CAF_MAIN()
