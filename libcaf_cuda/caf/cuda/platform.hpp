#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

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

  static platform_ptr create(); //singleton method that creates or gets platform


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

  // Resets the CUDA context for a specific device.
  // This will destroy the existing context for the device, create a new one,
  // and update the device object and platform's internal records.
  void reset_device_context(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
      throw std::out_of_range("Invalid device_id for reset_device_context");
    }

    device_ptr dev_obj = devices_[device_id];
    CUdevice cu_dev = dev_obj->getDevice(); // Get the underlying CUdevice

    // Create a new context
    CUcontext new_ctx;
#if CUDA_VERSION >= 13000
    {
      CUctxCreateParams ctx_params = {};
      CHECK_CUDA(cuCtxCreate(&new_ctx, 
                             &ctx_params, 
                             CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, 
                             cu_dev));
    }
#else
    CHECK_CUDA(cuCtxCreate(&new_ctx, 
                           CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, 
                           cu_dev));
#endif

    // Update the device object and platform's internal contexts_ vector
    dev_obj->reset_context(new_ctx);
    contexts_[device_id] = new_ctx;
  }

  //returns how many devices are currently on the GPU
  int get_num_devices();


private:
  platform();
  ~platform();

  std::string name_;
  std::string vendor_;
  std::string version_;
  std::vector<device_ptr> devices_;
  std::vector<CUcontext> contexts_;
  std::unique_ptr<scheduler> scheduler_;
};

// Intrusive pointer hooks
inline void intrusive_ptr_add_ref(platform* p) { p->ref(); }
inline void intrusive_ptr_release(platform* p) { p->deref(); }

} // namespace caf::cuda
