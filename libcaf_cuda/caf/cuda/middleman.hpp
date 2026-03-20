#pragma once

#include <string>
#include <vector>
#include <mutex>

#include <caf/actor_system.hpp>
#include <caf/actor_system_module.hpp>
#include <caf/actor.hpp>
#include <caf/intrusive_ptr.hpp>
#include <caf/version.hpp>

#include "caf/cuda/global_export.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/manager_config.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include "caf/detail/spawn_helper.hpp"

namespace caf::cuda {

class device;
using device_ptr = caf::intrusive_ptr<device>;

class program;
using program_ptr = caf::intrusive_ptr<program>;

class platform;
using platform_ptr = caf::intrusive_ptr<platform>;

template <bool PassConfig, class... Ts>
class actor_facade;

/// A proper CAF module that manages CUDA resources.
/// Replaces the former global singleton `manager`.
///
/// Usage:
/// @code
/// void caf_main(caf::actor_system& sys) {
///   auto& cuda = sys.cuda();
///   auto prog = cuda.create_program_from_cubin("kernel.cubin", "myKernel");
///   // ...
/// }
/// CAF_MAIN(caf::cuda::middleman)
/// @endcode
class CAF_CUDA_EXPORT middleman : public actor_system_module {
public:
  friend class actor_system;

  // -- static module interface (required by CAF_MAIN / cfg.load<T>()) ---------

  /// Registers CUDA type IDs in the global meta object table.
  static void init_global_meta_objects();

  /// Adds CUDA-specific options to the actor system config.
  static void add_module_options(actor_system_config& cfg);

  /// Factory method called by the actor system to create this module.
  static actor_system_module* make(actor_system& sys);

  /// Checks ABI compatibility with the CAF core library.
  static void check_abi_compatibility(version::abi_token token);

  // -- constructors / destructors ---------------------------------------------

  ~middleman() override;

  // -- actor_system_module interface ------------------------------------------

  void start() override;
  void stop() override;
  void init(actor_system_config& cfg) override;
  id_t id() const override;
  void* subtype_ptr() override;

  // -- public API (replaces manager::get().method()) --------------------------

  /// Returns a reference to the host actor system.
  actor_system& system() { return system_; }

  /// Finds a device by its index.
  device_ptr find_device(int id);

  /// Returns the number of available CUDA devices.
  int get_num_devices();

  /// Creates a program from a CUDA source string using NVRTC.
  program_ptr create_program(const char* kernel,
                             const std::string& name,
                             device_ptr dev);

  /// Creates a program from a file on disk using NVRTC.
  program_ptr create_program_from_file(const std::string& filename,
                                       const char* options,
                                       device_ptr dev);

  /// Creates a program from a precompiled CUBIN file.
  program_ptr create_program_from_cubin(const std::string& filename,
                                        const char* kernel_name,
                                        device_ptr device);

  /// Creates a program from a precompiled CUBIN file (auto-selects device).
  program_ptr create_program_from_cubin(const std::string& filename,
                                        const char* kernel_name);

  /// Creates a program from a precompiled FATBIN file.
  program_ptr create_program_from_fatbin(const std::string& filename,
                                         const char* kernel_name);

  /// Spawns a GPU actor facade from a CUDA source string.
  template <class... Ts>
  caf::actor spawn(const char* kernel,
                   const std::string& name,
                   nd_range dims,
                   Ts&&... xs) {
    caf::detail::cuda_spawn_helper<false, Ts...> f;
    caf::actor_config cfg;
    device_ptr device = find_device(0);
    program_ptr prog = create_program(kernel, name, device);
    return f(&system_, std::move(cfg), std::move(prog), dims,
             std::forward<Ts>(xs)...);
  }

  /// Spawns a GPU actor facade from a precompiled CUBIN file.
  template <class... Ts>
  caf::actor spawnFromCUBIN(const std::string& fileName,
                            const char* kernelName,
                            nd_range dims,
                            Ts&&... xs) {
    caf::detail::cuda_spawn_helper<false, Ts...> f;
    caf::actor_config cfg;
    device_ptr device = find_device(0);
    program_ptr prog = create_program_from_cubin(fileName, kernelName, device);
    return f(&system_, std::move(cfg), std::move(prog), dims,
             std::forward<Ts>(xs)...);
  }

  /// Returns the scheduler actor for the given device (or default).
  caf::actor get_scheduler_actor();

  /// Sends a token to a scheduler actor.
  void send_scheduler_actor_message(token_ptr token, int device_number = -1);
  void send_scheduler_actor_message(std::vector<token_ptr> tokens,
                                    int device_number = -1);
  void send_scheduler_actor_message(behavior_token_ptr token,
                                    int device_number);
  void send_scheduler_actor_message(std::string behavior, int device_number);

  /// Returns the memory management actor.
  caf::actor get_memory_actor();

  /// Spawns an exit-coordination actor.
  caf::actor spawn_exit_actor(int num_actors);

private:
  explicit middleman(actor_system& sys);

  /// Compiles a CUDA source string to PTX for a specific device.
  bool compile_nvrtc_program(const char* source, CUdevice device,
                             std::vector<char>& ptx_out);

  /// Spawns one scheduler actor per device.
  void init_scheduler_actors();

  /// Spawns the memory management actor.
  void init_memory_actor();

  /// Destroys the memory management actor.
  void destroy_memory_actor();

  actor_system& system_;
  platform_ptr platform_;
  bool scheduler_on_ = false;
  bool memory_manager_on_ = false;
  caf::actor memory_actor_handle_;
  std::vector<caf::actor> scheduler_actors_;
};

} // namespace caf::cuda
