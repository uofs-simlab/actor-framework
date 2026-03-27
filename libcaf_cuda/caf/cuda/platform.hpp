#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda.h>

#include <caf/intrusive_ptr.hpp>
#include <caf/actor_system.hpp>

#include "caf/ref_counted.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/scheduler.hpp"

namespace caf::cuda {

// Forward-declared implementation detail; defined in platform.cpp.
// Holds a background thread that defers cuModuleUnload calls so that
// program::~program() never blocks a CAF worker thread.
class deferred_module_unloader;

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

  /// Enqueues a (ctx, module) pair for deferred cleanup on the background
  /// thread.  The thread will cuCtxSynchronize then cuModuleUnload so that
  /// ~program() never blocks the calling thread.
  void defer_module_unload(CUcontext ctx, CUmodule module);

  std::string name_;

private:
  platform();
  ~platform();

  std::string vendor_;
  std::string version_;
  std::vector<device_ptr> devices_;
  std::vector<CUcontext> contexts_;
  std::unique_ptr<scheduler> scheduler_;
  // Declared last → destroyed first (before devices_/contexts_).
  // The destructor joins the background thread, draining all pending unloads
  // before device contexts are torn down.
  std::unique_ptr<deferred_module_unloader> module_cleanup_;
};

// Intrusive pointer hooks
inline void intrusive_ptr_add_ref(platform* p) { p->ref(); }
inline void intrusive_ptr_release(platform* p) { p->deref(); }

} // namespace caf::cuda

