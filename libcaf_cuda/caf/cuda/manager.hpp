#pragma once

#include <string>
#include <cstddef>
#include <stdexcept>
#include <mutex>
#include <fstream>
#include <map>
#include <shared_mutex>
#include <unordered_map>

#include <caf/actor_system.hpp>
#include <caf/actor.hpp>
#include <caf/intrusive_ptr.hpp>
#include <caf/detail/type_list.hpp>
#include <caf/init_global_meta_objects.hpp>

#include "caf/detail/spawn_helper.hpp"

#include "caf/cuda/global.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/actor_facade.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/manager_config.hpp"
#include "caf/cuda/control-layer/token.hpp" // For send_scheduler_actor_message
#include "caf/cuda/control-layer/behavior_token.hpp"


//A class that just acts as a user interface
//and a system initialization for cuda 
namespace caf::cuda {

class device;
using device_ptr = caf::intrusive_ptr<device>;

class program;
using program_ptr = caf::intrusive_ptr<program>;

class platform;
using platform_ptr = caf::intrusive_ptr<platform>;

template <bool PassConfig, class... Ts>
class actor_facade;

class CAF_CUDA_EXPORT manager {
public:
  /// Initializes the singleton. Must be called exactly once before get().
  static void init(caf::actor_system& sys);




 /// Initializes the singleton. Must be called exactly once before get().
  static void init(caf::actor_system& sys,manager_config config);

  /// Returns the singleton instance. Crashes if not yet initialized.
  static manager& get();

  /// Deletes the singleton if needed (optional).
  //deletes the scheduler actor as well if it exists
  static void shutdown();

  /// Flushes the cache of compiled programs.
  void flush_programs();

  // Prevent copy/assignment
  manager(const manager&) = delete;
  manager& operator=(const manager&) = delete;

  //device_ptr getter
  device_ptr find_device(std::size_t id) const;


  //Creates a program ptr to be used to launch kernels
  //@param: kernel, A string representation of a kernel
  //@param:name, the function signature name of the kernel
  //@param dev, device pointer
  program_ptr create_program(const char* kernel,
                             const std::string& name,
                             device_ptr dev);


  program_ptr create_program_from_file(const std::string& filename,
                                       const char* options,
                                       device_ptr dev);


  //Currently not working DO NOT USE 
  program_ptr create_program_from_ptx(const std::string& filename,
                                    const char* kernel_name,
                                    device_ptr device);

  program_ptr create_program_from_cubin(const std::string& filename,
                                               const char* kernel_name,
                                               device_ptr device);



 //Creates a program ptr to be used to launch kernels
 //must read in a precompiled cubin file 
  //@param: filename, this is the path to the file that contains the kernel
  //@param: kernel_name, the function signature name of the kernel
  program_ptr create_program_from_cubin(const std::string& filename,
                                               const char* kernel_name);



 //Creates a program ptr to be used to launch kernels
 //must read in a precompiled fatbin file 
  //@param: filename, this is the path to the file that contains the kernel
  //@param: kernel_name, the function signature name of the kernel
  program_ptr create_program_from_fatbin(const std::string& filename,
                                               const char* kernel_name);



  //Spawns in an actor facade 
  //@param, kernel, string version of the kernel
  //@param name, name of the function that is the kernel 
  //@param dims: both block and grid dimensions of the kernel you want to launch
  //@returns a handle to the actor facade
  template <class... Ts>
  caf::actor spawn(const char* kernel,
                   const std::string& name,
		   nd_range dims,
                   Ts&&... xs) {
    caf::detail::cuda_spawn_helper<false, Ts...> f;
    caf::actor_config cfg;

    device_ptr device = find_device(0);
    program_ptr prog = create_program(kernel, name, device);

    return f(&system_, std::move(cfg), std::move(prog),dims,std::forward<Ts>(xs)...);
  }


  //Currently broken DO not use  
  template <class... Ts>
  caf::actor spawnFromPTX(
                   const std::string& fileName,
		   const char * kernelName,
		   nd_range dims,
                   Ts&&... xs) {
    caf::detail::cuda_spawn_helper<false, Ts...> f;
    caf::actor_config cfg;

    device_ptr device = find_device(0);
    program_ptr prog = create_program_from_ptx(fileName, kernelName, device);

    return f(&system_, std::move(cfg), std::move(prog),dims,std::forward<Ts>(xs)...);
  }



 
  //Spawns an actor in from a precompiled cubin
  //must read in a precompiled cubin file
  //@param filename, path to the file
  //@param kernelName: the name of the kernel
  //@param dims: both block and grid dimensions of the kernel you want to launch
  //@returns a handle to the actor facade
  template <class... Ts>
  caf::actor spawnFromCUBIN(
                   const std::string& fileName,
		   const char * kernelName,
		   nd_range dims,
                   Ts&&... xs) {
    caf::detail::cuda_spawn_helper<false, Ts...> f;
    caf::actor_config cfg;

    device_ptr device = find_device(0);
    program_ptr prog = create_program_from_cubin(fileName, kernelName, device);

    return f(&system_, std::move(cfg), std::move(prog),dims,std::forward<Ts>(xs)...);
  }


  caf::actor_system& system() { return system_; }

  device_ptr find_device(int id);

  int get_num_devices();

  double available_memory_mb(int id = 0);

  caf::actor spawn_exit_actor(int num_actors);

  // Toggles the scheduler actors on. If called multiple times, it will only spawn them once.
  void toggle_scheduler_actor(int num_streams, int stream_depth);

  // Enables cuBLAS support on all detected devices.
  void enable_blas_actors();

  // Enables cuSparse support on all detected devices.
  void enable_sparse_actors();

  // Sends a batch of tokens to the scheduler actors, distributing them statically.
  void send_scheduler_actor_message(std::vector<token_ptr> tokens);

  // Sends a single token to a randomly selected scheduler actor.
  void send_scheduler_actor_message(token_ptr token);

  // Returns the first scheduler actor. Useful for single-GPU setups or general dispatch.
  caf::actor get_scheduler_actor();

  // Returns a specific scheduler actor by device number.
  caf::actor get_scheduler_actor(int device_number);

  // Sends a behavior change message to a specific scheduler actor.
  void send_scheduler_actor_message(const std::string& behavior_name, int device_number);

private:
  explicit manager(caf::actor_system& sys)
    : system_(sys), platform_(platform::create()) {
    // cuInit is done in init()
  }

  caf::actor_system& system_;
  platform_ptr platform_;

  //helper to compile a nvrtc program
  bool compile_nvrtc_program(const char* source, CUdevice device, std::vector<char>& ptx_out);

  static manager* instance_;
  static std::mutex mutex_;

  mutable std::shared_mutex programs_mutex_;
  std::unordered_map<size_t, program_ptr> programs_;
  std::vector<caf::actor> scheduler_actors_; // Stores handles to spawned scheduler actors
  bool scheduler_actors_spawned_ = false; // Flag to ensure idempotency
};

} // namespace caf::cuda
