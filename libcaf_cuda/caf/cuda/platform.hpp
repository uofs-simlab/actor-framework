#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <utility>
#include <cuda.h>

#include <caf/intrusive_ptr.hpp>
#include <caf/actor_system.hpp>

#include "caf/ref_counted.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/scheduler.hpp"

namespace caf::cuda {

//A container that has access to all the devices and a scheduler 
//to select which device an actor should go onto
//is An actors first point of contact when they want to access the device
//Actors are not allowed to have any access to the device class 
class CAF_CUDA_EXPORT platform : public ref_counted {
public:
  friend class program;
  template <class T, class... Ts>
  friend intrusive_ptr<T> caf::make_counted(Ts&&...);


  const std::string& name() const; 
  const std::string& vendor() const;
  const std::string& version() const;
  //returns a list of devices 
  const std::vector<device_ptr>& devices() const;

  //returns a single device given its id
  device_ptr getDevice(int id);
  //returns the scheduler being used 
  scheduler* get_scheduler();

  //returns the device that a command should use
  device_ptr schedule(int actor_id);
  //returns the device that a command should used 
  device_ptr schedule(int actor_id,int device_number);
  //releases a stream for an actor
  void release_streams_for_actor(int actor_id);

  //returns how many devices are currently on the GPU
  int get_num_devices();

  /// Records a (ctx, module) pair for deferred cleanup at platform shutdown.
  /// No synchronisation or unloading happens here — ~program() returns
  /// instantly and never blocks a CAF worker thread.
  void defer_module_unload(CUcontext ctx, CUmodule module);

  /// Immediately synchronises and unloads all retired modules that have been
  /// accumulated via defer_module_unload().  This is an optional escape valve
  /// for applications that dynamically load many different kernels during a
  /// long run and want to reclaim the (small) GPU memory occupied by module
  /// code without waiting for platform shutdown.
  ///
  /// **Warning:** this calls cuCtxSynchronize + cuModuleUnload for every
  /// retired module, so it will block until all in-flight GPU work completes.
  /// Only call this from a point where blocking is acceptable (e.g. between
  /// computation phases, never from inside a CAF actor handler).
  void flush_retired_modules();

  std::string name_;

private:
  platform();
  ~platform();

  std::string vendor_;
  std::string version_;
  std::vector<device_ptr> devices_;
  std::vector<CUcontext> contexts_;
  std::unique_ptr<scheduler> scheduler_;
  // Retired modules accumulated by defer_module_unload().  Unloaded either
  // by an explicit flush_retired_modules() call or in ~platform() when
  // all GPU work is guaranteed complete (cuCtxDestroy handles the rest).
  std::mutex                                        retired_modules_mutex_;
  std::vector<std::pair<CUcontext, CUmodule>>        retired_modules_;
};

// Intrusive pointer hooks
inline void intrusive_ptr_add_ref(platform* p) { p->ref(); }
inline void intrusive_ptr_release(platform* p) { p->deref(); }

} // namespace caf::cuda

