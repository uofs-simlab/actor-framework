#include "caf/cuda/platform.hpp"
#include <iostream>
#include <mutex>

namespace caf::cuda {

//constructor
platform::platform() {
  int device_count = 0;
  check(cuDeviceGetCount(&device_count));
  devices_.resize(device_count);
  contexts_.resize(device_count);


  std::vector<std::string> device_names(device_count);

  for (int i = 0; i < device_count; ++i) {
    CUdevice cuda_device;
    check(cuDeviceGet(&cuda_device, i));

    char name[256];
    check(cuDeviceGetName(name, sizeof(name), cuda_device));
    device_names[i] = name;

#if CUDA_VERSION >= 13000
    {
      CUctxCreateParams ctx_params = {};
      check(cuCtxCreate(&contexts_[i], 
                        &ctx_params, 
                        CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, 
                        cuda_device));
    }
#else
    check(cuCtxCreate(&contexts_[i], 
                      CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, 
                      cuda_device));
#endif
    devices_[i] = make_counted<device>(cuda_device, contexts_[i], name, i);
  }

  // Check if all devices are the same by comparing their names
  bool all_same = std::all_of(
    device_names.begin() + 1,
    device_names.end(),
    [&](const std::string& name) { return name == device_names[0]; });

  //as of right now the multi gpu scheduler cannot handle 
  //devices that are not the same so if this is detected 
  //we turn off multi gpu support
  if (device_count > 1 && all_same) {
    scheduler_ = std::make_unique<multi_device_scheduler>();
  } else {
    scheduler_ = std::make_unique<single_device_scheduler>();
  }

  scheduler_->set_devices(devices_);

  if (device_count > 0) {
    check(cuCtxSetCurrent(contexts_[0]));
  }
}


platform::~platform() {
  // Unload any remaining retired modules before device contexts are
  // destroyed.  At this point the actor system is shut down, all actors are
  // dead, and no GPU work should be in-flight.
  std::cout << "CAF-CUDA platform shutting down, unloading retired modules..." << std::endl;
  flush_retired_modules();
}

void platform::defer_module_unload(CUcontext ctx, CUmodule module) {
  std::lock_guard<std::mutex> lk(retired_modules_mutex_);
  retired_modules_.emplace_back(ctx, module);
}

void platform::flush_retired_modules() {
  std::vector<std::pair<CUcontext, CUmodule>> modules;
  {
    std::lock_guard<std::mutex> lk(retired_modules_mutex_);
    modules.swap(retired_modules_);
  }
  for (auto& [ctx, mod] : modules) {
    CUresult res = cuCtxPushCurrent(ctx);
    if (res == CUDA_SUCCESS) {
      cuCtxSynchronize();
      cuModuleUnload(mod);
      CUcontext dummy;
      cuCtxPopCurrent(&dummy);
    }
    // If cuCtxPushCurrent failed the context is already gone;
    // the module was cleaned up with it.
  }
}

const std::string& platform::name() const {
  return name_;
}

const std::string& platform::vendor() const {
  return vendor_;
}

const std::string& platform::version() const {
  return version_;
}

const std::vector<device_ptr>& platform::devices() const {
  return devices_;
}

device_ptr platform::getDevice(int id) {
  if (id < 0 || static_cast<size_t>(id) >= devices_.size()) {
    throw std::out_of_range("Invalid device ID");
  }
  return devices_[id];
}

int platform::get_num_devices() { return devices_.size();}

scheduler* platform::get_scheduler() {
  return scheduler_.get();
}

device_ptr platform::schedule(int actor_id) {
  
	return scheduler_->schedule(actor_id);
}

device_ptr platform::schedule(int actor_id,int device_number) {
  
	return scheduler_->schedule(actor_id,device_number);
}


void platform::release_streams_for_actor(int actor_id) {
  for (auto& dev : devices_) {
    dev->release_stream_for_actor(actor_id);
  }
}

} // namespace caf::cuda

