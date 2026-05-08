#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <numeric>

using namespace caf;

CAF_BEGIN_TYPE_ID_BLOCK(memory_stress, caf::id_block::cuda::end)
  CAF_ADD_ATOM(memory_stress, start_atom)
  CAF_ADD_ATOM(memory_stress, task_done_atom)
CAF_END_TYPE_ID_BLOCK(memory_stress)

using mmulCommand = caf::cuda::command_runner<in<int>, 
                                              in<int>, 
                                              out<int>, 
                                              in<int>>;
mmulCommand mmul;

// Pre-allocated static data to avoid CPU overhead during the test
static std::vector<int> GLOBAL_A;
static std::vector<int> GLOBAL_B;

struct worker_state {
  static inline const char* name = "stress_worker";
  int actor_id = rand();
  caf::cuda::program_ptr program;
};

caf::behavior stress_worker_fun(caf::stateful_actor<worker_state>* self, caf::actor supervisor) {
  // Load the program once during actor initialization
  auto& mgr = caf::cuda::manager::get();
  self->state().program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

  return {
    [=](int N) {
      const int THREADS = 32;
      const int BLOCKS = (N + THREADS - 1) / THREADS;
      caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

      // We use the pre-allocated GLOBAL vectors. 
      auto results = mmul.run_async(self->state().program, dims, self->state().actor_id,
                                    caf::cuda::create_in_arg(GLOBAL_A),
                                    caf::cuda::create_in_arg(GLOBAL_B),
                                    caf::cuda::create_out_arg(N * N),
                                    caf::cuda::create_in_arg(N));

      auto matrixC_ptr = std::get<2>(results);

      // USER REQUEST: Do NOT capture matrixC_ptr in the lambda.
      // This will allow the mem_ptr to be destroyed immediately after this call.
      mmul.copy_to_host_async(matrixC_ptr, [supervisor](std::vector<int> h_c) mutable {
        caf::anon_mail(task_done_atom_v).send(supervisor);
      });
    }
  };
}

struct supervisor_state {
  int total_tasks = 2000;
  int tasks_submitted = 0;
  int tasks_completed = 0;
  int max_in_flight = 20; 
  caf::actor worker;
  size_t initial_mem = 0;
};

caf::behavior supervisor_fun(caf::stateful_actor<supervisor_state>* self) {
  auto submit_work = [=] {
    while (self->state().tasks_submitted < self->state().total_tasks &&
           (self->state().tasks_submitted - self->state().tasks_completed) < self->state().max_in_flight) {
      self->mail(1024).send(self->state().worker);
      self->state().tasks_submitted++;
    }
  };

  return {
    [=](start_atom) {
      auto& mgr_ref = caf::cuda::manager::get();
      self->state().initial_mem = mgr_ref.available_memory_mb();
      
      std::cout << "--- GPU Memory Stress Test Initialized ---" << std::endl;
      std::cout << "Initial Free Memory: " << self->state().initial_mem << " MB" << std::endl;
      std::cout << "Target Workload: " << self->state().total_tasks << " tasks" << std::endl;

      self->state().worker = self->spawn(stress_worker_fun, caf::actor_cast<caf::actor>(self));
      submit_work();
    },
    [=](task_done_atom) {
      self->state().tasks_completed++;

      if (self->state().tasks_completed % 100 == 0) {
        auto current_free = caf::cuda::manager::get().available_memory_mb();
        std::cout << "Progress: " << self->state().tasks_completed << "/" << self->state().total_tasks
                  << " | Current Free GPU Memory: " << current_free << " MB" << std::endl;
      }

      if (self->state().tasks_completed == self->state().total_tasks) {
        std::cout << "Test Finished. Final Free Memory: " << caf::cuda::manager::get().available_memory_mb() << " MB" << std::endl;
        self->send_exit(self->state().worker, exit_reason::user_shutdown);
        self->quit();
      } else {
        submit_work();
      }
    }
  };
}

void setup_global_data(int N) {
  GLOBAL_A.resize(N * N);
  GLOBAL_B.resize(N * N);
  std::iota(GLOBAL_A.begin(), GLOBAL_A.end(), 1);
  std::iota(GLOBAL_B.begin(), GLOBAL_B.end(), 1);
}

void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);
  setup_global_data(1024);
  
  auto sv = sys.spawn(supervisor_fun);
  caf::anon_mail(start_atom_v).send(sv);
  
  sys.await_all_actors_done();
}

CAF_MAIN(id_block::memory_stress)
