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
#include <unordered_map>
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
// specialized command runner used for scheduler-mode memory operations:
caf::cuda::command_runner<caf::cuda::mem_ptr<int>,
                         caf::cuda::mem_ptr<int>,
                         caf::cuda::mem_ptr<int>,
                         caf::cuda::mem_ptr<int>> mmul;

struct mmul_state {
  caf::cuda::program_ptr mmul_kernel;
};

struct mmul_actor_with_scheduler_state {
  static inline const char* name = "my_actor";
};

// --------------------------- Global matrix pools (indexed access) ---------------------------
// Key: matrix size N -> pool of shared_ptr<vector<int>> for that N.
// Actors will look up matrices via g_pools[N] and receive only small integer indices in messages.
static std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>>> g_pools;
static std::mutex g_pools_mutex;

// --------------------------- Deterministic matrix generation & pools ---------------------------
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

// --------------------------- Actor implementations (indexed pool, pool looked up from global) ---------------------------

caf::behavior mmul_actor_indexed(caf::stateful_actor<mmul_state>* self,
                                 caf::cuda::program_ptr mmul_kernel,
                                 int indexA,
                                 int indexB,
                                 int N) {
  // store program ptr
  self->state().mmul_kernel = mmul_kernel;
  // enqueue work to self (do not send pool pointer — actors will look up global pool by N)
  self->mail(indexA, indexB, N).send(self);

  return {
    [=](int idxA, int idxB, int N) {
      caf::cuda::manager& mgr = caf::cuda::manager::get();
      int device = 0;
      int stream = 0;

      const int THREADS = 32;
      const int BLOCKS = (N + THREADS - 1) / THREADS;
      caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

      auto program = self->state().mmul_kernel;

      // lookup pool globally
      std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>> pool;
      {
        std::lock_guard<std::mutex> lk(g_pools_mutex);
        auto it = g_pools.find(N);
        if (it == g_pools.end()) {
          std::cout  << "ERROR: no pool for N=" << N << "\n";
          self->quit();
          return;
        }
        pool = it->second;
      }

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
    int indexB) {

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
        // send the actual work to self (only indices and token, no pool pointer)
        self->mail(indexA, indexB, res_token, N).send(self);
      }
    },

    [=](int idxA,
        int idxB,
        const caf::cuda::response_token_ptr& res_token,
        int N) {

      // lookup pool globally
      std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>> pool;
      {
        std::lock_guard<std::mutex> lk(g_pools_mutex);
        auto it = g_pools.find(N);
        if (it == g_pools.end()) {
	  std::cout << "ERROR: no pool for N=" << N << "\n";
          self->quit();
          return;
        }
        pool = it->second;
      }

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

// --------------------------- Spawn helpers (take spawn_times) ---------------------------

void spawn_mmul_actors_with_schedule_scheduler(
    caf::actor_system& sys,
    const std::vector<double>& spawn_times,
    caf::actor exit_actor,
    int N,
    caf::cuda::program_ptr program,
    caf::cuda::nd_range dims,
    uint64_t master_seed) {

  // pool must already exist in g_pools[N]
  int pool_sz = 0;
  {
    std::lock_guard<std::mutex> lk(g_pools_mutex);
    auto it = g_pools.find(N);
    if (it == g_pools.end()) return;
    pool_sz = (int)it->second->size();
  }

  auto spawn_cb = [&](int actor_idx) {
    auto inds = choose_matrix_indices_for_actor(actor_idx, pool_sz, master_seed, N);
    int idxA = inds.first;
    int idxB = inds.second;
    // To avoid ambiguity with CAF spawn overloads we call the spawn via a lambda:
    sys.spawn([=](caf::stateful_actor<mmul_actor_with_scheduler_state>* s) -> caf::behavior {
        return mmul_actor_scheduler_indexed(s, exit_actor, N, program, dims, idxA, idxB);
    });
  };

  spawn_actors_with_schedule(sys, spawn_times, spawn_cb);
}

void spawn_mmul_actors_with_schedule_no_scheduler_actor(
    caf::actor_system& sys,
    const std::vector<double>& spawn_times,
    int N,
    caf::cuda::program_ptr program,
    uint64_t master_seed) {

  int pool_sz = 0;
  {
    std::lock_guard<std::mutex> lk(g_pools_mutex);
    auto it = g_pools.find(N);
    if (it == g_pools.end()) return;
    pool_sz = (int)it->second->size();
  }

  auto spawn_cb = [&](int actor_idx) {
    auto inds = choose_matrix_indices_for_actor(actor_idx, pool_sz, master_seed, N);
    int idxA = inds.first;
    int idxB = inds.second;

    // spawn mmul_actor_indexed using a lambda to bind arguments (avoids CAF type registration issues)
    sys.spawn([=](caf::stateful_actor<mmul_state>* s) -> caf::behavior {
      return mmul_actor_indexed(s, program, idxA, idxB, N);
    });
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

  std::cout << "=== MMUL Scaling Tests ===\n";
  std::cout << "Format:\n";
  std::cout << "scheduler matrix_size actors time_seconds\n";

  uint64_t master_seed = 0xDEADBEEF1234ULL;
  auto pools_map = prepare_matrix_pools(matrix_sizes, /*pool_size_per_size=*/32, master_seed);

  // move pools into global map so actors can read them by N without sending them in messages
  {
    std::lock_guard<std::mutex> lk(g_pools_mutex);
    for (auto& kv : pools_map) {
      int N = kv.first;
      auto vec = std::move(kv.second);
      g_pools[N] = std::make_shared<std::vector<std::shared_ptr<std::vector<int>>>>(std::move(vec));
    }
  }

  for (int size : matrix_sizes) {
    for (int actors : actor_counts) {

      // Prepare deterministic spawn_times once per (size, actors)
      double total_duration = 5.0; // seconds (tunable)
      int max_waves = 6;
      auto spawn_times = generate_spawn_schedule(actors, total_duration, max_waves, master_seed, size);

      // ================= Scheduler-enabled (core_usage) =================
      caf::cuda::manager::init(sys, man_config);
      {
        caf::cuda::manager& mgr = caf::cuda::manager::get();
        // set scheduler behavior per device
        for (int i = 0; i < mgr.get_num_devices(); i++)
          mgr.send_scheduler_actor_message("multilevel", i);

        std::cout << "[RUN] scheduler=multilevel_usage "
                  << "matrix_size=" << size << " "
                  << "actors=" << actors << "\n";

        auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
        const int THREADS = 32;
        const int BLOCKS = (size + THREADS - 1) / THREADS;
        caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

        caf::actor exit_actor = mgr.spawn_exit_actor(actors);

        double core_usage_time = time_run([&] {
          spawn_mmul_actors_with_schedule_scheduler(sys, spawn_times, exit_actor, size, program, dims, master_seed);
          sys.await_all_actors_done();
        });

        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT core_usage "
                  << size << " "
                  << actors << " "
                  << core_usage_time << "\n";
      }
      caf::cuda::manager::shutdown();

      // ================= Scheduler-disabled actor (green-light) =================
      caf::cuda::manager::init(sys, man_config);
      {
        caf::cuda::manager& mgr = caf::cuda::manager::get();
        for (int i = 0; i < mgr.get_num_devices(); i++)
          mgr.send_scheduler_actor_message("green", i);

        std::cout << "[RUN] scheduler=green_light_only "
                  << "matrix_size=" << size << " "
                  << "actors=" << actors << "\n";

        auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
        const int THREADS = 32;
        const int BLOCKS = (size + THREADS - 1) / THREADS;
        caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

        caf::actor exit_actor = mgr.spawn_exit_actor(actors);

        double green_light_time = time_run([&] {
          // reuse the same spawn_times and global pools
          spawn_mmul_actors_with_schedule_scheduler(sys, spawn_times, exit_actor, size, program, dims, master_seed);
          sys.await_all_actors_done();
        });

        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT green_light_only "
                  << size << " "
                  << actors << " "
                  << green_light_time << "\n";
      }
      caf::cuda::manager::shutdown();

      // ================= No scheduler at all actor =================
      caf::cuda::manager_config no_sched_config(false);
      caf::cuda::manager::init(sys, no_sched_config);
      {
        caf::cuda::manager& mgr = caf::cuda::manager::get();

        std::cout << "[RUN] scheduler=none "
                  << "matrix_size=" << size << " "
                  << "actors=" << actors << "\n";

        auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
        const int THREADS = 32;
        const int BLOCKS = (size + THREADS - 1) / THREADS;
        caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

        double no_scheduler_time = time_run([&] {
          spawn_mmul_actors_with_schedule_no_scheduler_actor(sys, spawn_times, size, program, master_seed);
          sys.await_all_actors_done();
        });

        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT none "
                  << size << " "
                  << actors << " "
                  << no_scheduler_time << "\n";
      }
      caf::cuda::manager::shutdown();
    }
  }

  std::cout << "=== MMUL Scaling Tests Complete ===\n";
}

void caf_main(caf::actor_system& sys) {
  caf::cuda::manager_config man_config(true);
  run_mmul_scaling_tests(sys, man_config);
}

CAF_MAIN()
