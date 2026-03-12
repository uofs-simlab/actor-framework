#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/component-actors/all-component-actors.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <random>
#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include "caf/actor_registry.hpp"

using namespace caf;
using namespace std::chrono_literals;

// --------------------------- Types / globals ---------------------------
using command = caf::cuda::command_runner<>;
command mmul_command;
using async_command = caf::cuda::mmul_async_command<int>;
async_command async_mmul;
caf::cuda::command_runner<caf::cuda::mem_ptr<int>, caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int >> mmul;


struct mmul_state {
  caf::cuda::program_ptr mmul_kernel;
};

struct mmul_actor_with_scheduler_state {
  static inline const char* name = "my_actor";
};

// --------------------------- Deterministic matrix generation & pools (shared pool, pass indices) ---------------------------
static std::vector<int> generate_matrix(int N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> dist(0, 9);
  std::vector<int> mat;
  mat.resize((size_t)N * N);
  for (size_t i = 0; i < mat.size(); ++i)
    mat[i] = dist(rng);
  return mat;
}

// returns map<size, vector<shared_ptr<vector<int>>>>
static std::map<int, std::vector<std::shared_ptr<std::vector<int>>>> prepare_matrix_pools(
    const std::vector<int>& sizes,
    int pool_size_per_size,
    uint64_t master_seed) {

  std::map<int, std::vector<std::shared_ptr<std::vector<int>>>> pools;
  for (int N : sizes) {
    int pool_sz = pool_size_per_size;
    if (N >= 1024) pool_sz = std::min(pool_sz, 8);
    if (N >= 2048) pool_sz = std::min(pool_sz, 4);

    pools[N] = {};
    for (int i = 0; i < pool_sz; ++i) {
      uint64_t seed = master_seed ^ (uint64_t(N) << 32) ^ (uint64_t(i) * 0x9e3779b97f4a7c15ULL);
      auto v = std::make_shared<std::vector<int>>(generate_matrix(N, seed));
      pools[N].push_back(std::move(v));
    }
  }
  return pools;
}

// deterministic spawn schedule generator (waves model)
static std::vector<double> generate_spawn_schedule(
    int actors,
    double total_duration_seconds,
    int max_waves,
    uint64_t master_seed,
    int size /* used to vary seeds per trial */) {

  std::mt19937_64 rng(master_seed ^ uint64_t(size));
  std::uniform_real_distribution<double> time_dist(0.0, total_duration_seconds);
  std::uniform_int_distribution<int> waves_dist(1, std::max(1, max_waves));
  std::uniform_real_distribution<double> jitter_dist(0.0, 0.25 * total_duration_seconds);

  int num_waves = waves_dist(rng);
  std::vector<std::pair<double,int>> waves;
  std::vector<int> wave_sizes(num_waves, 0);
  int remaining = actors;
  for (int w = 0; w < num_waves; ++w) {
    if (w == num_waves - 1) {
      wave_sizes[w] = remaining;
    } else {
      int max_allowed = std::max(1, remaining - (num_waves - w - 1));
      std::uniform_int_distribution<int> size_dist(1, max_allowed);
      int chosen = size_dist(rng);
      wave_sizes[w] = chosen;
      remaining -= chosen;
    }
  }
  for (int w = 0; w < num_waves; ++w) {
    double t = time_dist(rng);
    waves.emplace_back(t, wave_sizes[w]);
  }

  std::vector<double> spawn_times;
  for (int w = 0; w < num_waves; ++w) {
    double wt = waves[w].first;
    int wsize = waves[w].second;
    for (int i = 0; i < wsize; ++i) {
      double jitter = jitter_dist(rng) * ((i % 2 == 0) ? 1.0 : -1.0);
      double st = wt + jitter;
      if (st < 0.0) st = 0.0;
      if (st > total_duration_seconds) st = total_duration_seconds;
      spawn_times.push_back(st);
    }
  }
  if ((int)spawn_times.size() > actors) {
    spawn_times.resize(actors);
  } else if ((int)spawn_times.size() < actors) {
    std::uniform_real_distribution<double> extra_dist(0.0, total_duration_seconds);
    while ((int)spawn_times.size() < actors)
      spawn_times.push_back(extra_dist(rng));
  }
  std::sort(spawn_times.begin(), spawn_times.end());
  return spawn_times;
}

// utility to pick indices deterministically for each actor
static std::pair<int,int> choose_matrix_indices_for_actor(int actor_index, int pool_size, uint64_t master_seed, int size) {
  std::mt19937_64 rng(master_seed ^ uint64_t(actor_index) ^ uint64_t(size << 16));
  std::uniform_int_distribution<int> idx_dist(0, std::max(0, pool_size - 1));
  int a = idx_dist(rng);
  int b = idx_dist(rng);
  return {a, b};
}

// spawn actors according to schedule (sleeps on caller thread)
static void spawn_actors_with_schedule(caf::actor_system& sys,
                                       const std::vector<double>& spawn_times,
                                       std::function<void(int)> spawn_cb) {
  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < spawn_times.size(); ++i) {
    double target = spawn_times[i];
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - t0).count();
    if (target > elapsed) {
      std::this_thread::sleep_for(std::chrono::duration<double>(target - elapsed));
    }
    spawn_cb(int(i));
  }
}

// --------------------------- Actor implementations (indexed pool) ---------------------------

caf::behavior mmul_actor_indexed(caf::stateful_actor<mmul_state>* self,
                                 caf::cuda::program_ptr mmul_kernel,
                                 int indexA,
                                 int indexB,
                                 int N,
                                 std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>> pool_ptr) {
  // store program ptr
  self->state().mmul_kernel = mmul_kernel;
  // enqueue work to self to keep consistency with message-driven design
  self->mail(indexA, indexB, N, pool_ptr).send(self);

  return {
    [=](int idxA, int idxB, int N, std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>> pool) {
      caf::cuda::manager& mgr = caf::cuda::manager::get();
      int device = 0;
      int stream = 0;

      const int THREADS = 32;
      const int BLOCKS = (N + THREADS - 1) / THREADS;
      caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

      auto program = self->state().mmul_kernel;

      auto matA_ptr = (*pool)[idxA];
      auto matB_ptr = (*pool)[idxB];

      auto inA = caf::cuda::create_in_arg(*matA_ptr);
      auto arg1 = mmul_command.transfer_memory(device, stream, std::move(inA));

      auto inB = caf::cuda::create_in_arg(*matB_ptr);
      auto arg2 = mmul_command.transfer_memory(device, stream, std::move(inB));

      auto arg3 = caf::cuda::create_out_arg<int>(N * N);
      auto arg4 = caf::cuda::create_in_arg<int>(N);

      auto result = async_mmul.run_async(program, dims, stream, 0, device, arg1, arg2, arg3, arg4);
      std::get<2>(result)->copy_to_host();
      self->quit();
    }
  };
}

caf::behavior mmul_actor_scheduler_indexed(
    caf::stateful_actor<mmul_actor_with_scheduler_state>* self,
    caf::actor exit_actor,
    int N,
    caf::cuda::program_ptr program,
    caf::cuda::nd_range dims,
    int indexA,
    int indexB,
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>> pool_ptr) {

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  // request a launch token from scheduler
  caf::cuda::token_ptr launch_token = caf::cuda::make_launch_token(
                    program,
                    dims,
                    0,
                    "mmul",
                    self,
                    rand());
  mgr.send_scheduler_actor_message(launch_token);

  // when we receive a launch response, scheduler will send a response_token which we handle below
  return {
    [=](caf::cuda::response_token_ptr res_token) {
      if (res_token->getType() == LAUNCH_RESPONSE) {
        // send the actual work to self
        self->mail(indexA, indexB, res_token, N, pool_ptr).send(self);
      }
    },

    [=](int idxA,
        int idxB,
        const caf::cuda::response_token_ptr& res_token,
        int N,
        std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>> pool) {

      auto matA_ptr = (*pool)[idxA];
      auto matB_ptr = (*pool)[idxB];

      auto arg1 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(*matA_ptr));
      auto arg2 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(*matB_ptr));
      auto arg3 = mmul.transfer_memory(res_token, caf::cuda::create_out_arg(N*N));
      auto arg4 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(N));

      auto tempC = mmul.run_async(program, dims, res_token, arg1, arg2, arg3, arg4);
      caf::cuda::mem_ptr<int> bufferC = std::get<2>(tempC);

      bufferC->synchronize();
      res_token->release();
      bufferC->copy_to_host();

      self->mail(1).send(exit_actor);
      self->quit();
    }
  };
}

// --------------------------- Spawn helpers (take spawn_times and shared pool_ptr) ---------------------------

void spawn_mmul_actors_with_schedule_scheduler(
    caf::actor_system& sys,
    const std::vector<double>& spawn_times,
    caf::actor exit_actor,
    int N,
    caf::cuda::program_ptr program,
    caf::cuda::nd_range dims,
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>> pool_ptr,
    uint64_t master_seed) {

  int pool_sz = (int)pool_ptr->size();
  auto spawn_cb = [&](int actor_idx) {
    auto inds = choose_matrix_indices_for_actor(actor_idx, pool_sz, master_seed, N);
    int idxA = inds.first;
    int idxB = inds.second;

    sys.spawn(
      mmul_actor_scheduler_indexed,
      exit_actor,
      N,
      program,
      dims,
      idxA,
      idxB,
      pool_ptr
    );
  };

  spawn_actors_with_schedule(sys, spawn_times, spawn_cb);
}

void spawn_mmul_actors_with_schedule_no_scheduler_actor(
    caf::actor_system& sys,
    const std::vector<double>& spawn_times,
    int N,
    caf::cuda::program_ptr program,
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>> pool_ptr,
    uint64_t master_seed) {

  int pool_sz = (int)pool_ptr->size();
  auto spawn_cb = [&](int actor_idx) {
    auto inds = choose_matrix_indices_for_actor(actor_idx, pool_sz, master_seed, N);
    int idxA = inds.first;
    int idxB = inds.second;

    sys.spawn(
      mmul_actor_indexed,
      program,
      idxA,
      idxB,
      N,
      pool_ptr
    );
  };

  spawn_actors_with_schedule(sys, spawn_times, spawn_cb);
}

// --------------------------- Test driver (integrates everything) ---------------------------

template <class Fn>
double time_run(Fn&& fn) {
  auto start = std::chrono::steady_clock::now();
  fn();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}

void run_mmul_scaling_tests(caf::actor_system& sys, caf::cuda::manager_config man_config) {
  const int max_size   = 2048;
  const int min_actors = 1;
  const int max_actors = 1024;

  std::vector<int> matrix_sizes = {10};
  for (int s = 32; s <= max_size; s *= 2)
    matrix_sizes.push_back(s);

  std::vector<int> actor_counts;
  for (int a = min_actors; a <= max_actors; a *= 2)
    actor_counts.push_back(a);

  std::cout << "=== MMUL Scaling Tests ===";
  std::cout << "Format:";
  std::cout << "scheduler matrix_size actors time_seconds";

  uint64_t master_seed = 0xDEADBEEF1234ULL;
  auto pools_map = prepare_matrix_pools(matrix_sizes, /*pool_size_per_size=*/32, master_seed);

  for (int size : matrix_sizes) {
    for (int actors : actor_counts) {

      // Prepare deterministic spawn_times once per (size, actors)
      double total_duration = 5.0; // seconds (tunable)
      int max_waves = 6;
      auto spawn_times = generate_spawn_schedule(actors, total_duration, max_waves, master_seed, size);

      // make a shared_ptr to the pool for this size so we can cheaply pass it to actors
      auto pool_vec = pools_map[size];
      auto pool_shared = std::make_shared<std::vector<std::shared_ptr<std::vector<int>>>>(std::move(pool_vec));

      // ================= Scheduler-enabled (core_usage) =================
      caf::cuda::manager::init(sys, man_config);
      {
        caf::cuda::manager& mgr = caf::cuda::manager::get();
        // set scheduler behavior per device
        for (int i = 0; i < mgr.get_num_devices(); i++)
          mgr.send_scheduler_actor_message("multilevel", i);

        std::cout << "[RUN] scheduler=multilevel_usage \n"
                  << "matrix_size=" << size
                  << " actors=" << actors << "";

        auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
        const int THREADS = 32;
        const int BLOCKS = (size + THREADS - 1) / THREADS;
        caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

        caf::actor exit_actor = mgr.spawn_exit_actor(actors);

        double core_usage_time = time_run([&] {
          spawn_mmul_actors_with_schedule_scheduler(sys, spawn_times, exit_actor, size, program, dims, pool_shared, master_seed);
          sys.await_all_actors_done();
        });

        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT core_usage \n"
                  << size << " \n"
                  << actors << " \n"
                  << core_usage_time << "\n";
      }
      caf::cuda::manager::shutdown();

      // ================= Scheduler-disabled actor (green-light) =================
      caf::cuda::manager::init(sys, man_config);
      {
        caf::cuda::manager& mgr = caf::cuda::manager::get();
        for (int i = 0; i < mgr.get_num_devices(); i++)
          mgr.send_scheduler_actor_message("green", i);

        std::cout << "[RUN] scheduler=green_light_only \n"
                  << "matrix_size=" << size
                  << " actors=" << actors << "\n";

        auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
        const int THREADS = 32;
        const int BLOCKS = (size + THREADS - 1) / THREADS;
        caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

        caf::actor exit_actor = mgr.spawn_exit_actor(actors);

        double green_light_time = time_run([&] {
          // reuse the same spawn_times and pool_shared
          spawn_mmul_actors_with_schedule_scheduler(sys, spawn_times, exit_actor, size, program, dims, pool_shared, master_seed);
          sys.await_all_actors_done();
        });

        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT green_light_only \n"
                  << size << " \n"
                  << actors << " \n"
                  << green_light_time << "\n";
      }
      caf::cuda::manager::shutdown();

      // ================= No scheduler at all actor =================
      caf::cuda::manager_config no_sched_config(false);
      caf::cuda::manager::init(sys, no_sched_config);
      {
        caf::cuda::manager& mgr = caf::cuda::manager::get();

        std::cout <<"[RUN] scheduler=none \n"
                  << "matrix_size=" << size
                  << " actors=" << actors << "\n";

        auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
        const int THREADS = 32;
        const int BLOCKS = (size + THREADS - 1) / THREADS;
        caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

        double no_scheduler_time = time_run([&] {
          spawn_mmul_actors_with_schedule_no_scheduler_actor(sys, spawn_times, size, program, pool_shared, master_seed);
          sys.await_all_actors_done();
        });

        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT none \n"
                  << size << " \n"
                  << actors << " \n"
                  << no_scheduler_time << "";
      }
      caf::cuda::manager::shutdown();
    }
  }

  std::cout << "=== MMUL Scaling Tests Complete ===";
}

void caf_main(caf::actor_system& sys) {
  caf::cuda::manager_config man_config(true);
  run_mmul_scaling_tests(sys, man_config);
}

CAF_MAIN()

