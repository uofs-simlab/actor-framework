#include "caf/cuda/platform.hpp"
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace caf::cuda {

// ---------------------------------------------------------------------------
// deferred_module_unloader
//
// A background thread whose sole job is to wait for GPU work to finish and
// then call cuModuleUnload.  By doing the synchronisation here rather than on
// a CAF worker thread, we prevent ~program() from ever blocking the actor
// scheduler.
// ---------------------------------------------------------------------------
class deferred_module_unloader {
  struct work_item {
    CUcontext ctx;
    CUmodule  module;
  };

  std::mutex              mutex_;
  std::queue<work_item>   queue_;
  std::condition_variable cv_;
  bool                    stop_ = false;
  std::thread             thread_;

  void run() {
    while (true) {
      std::unique_lock<std::mutex> lk(mutex_);
      cv_.wait(lk, [this] { return !queue_.empty() || stop_; });
      if (stop_ && queue_.empty()) return;
      auto item = queue_.front();
      queue_.pop();
      lk.unlock();

      // Push the device context onto this thread, synchronise to ensure the
      // module is no longer in use, then unload it safely.
      CUresult res = cuCtxPushCurrent(item.ctx);
      if (res == CUDA_SUCCESS) {
        // Ignore sync errors: the context may be shutting down gracefully.
        cuCtxSynchronize();
        cuModuleUnload(item.module);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
      }
      // If cuCtxPushCurrent failed the context (and module) are already gone;
      // nothing to clean up.
    }
  }

public:
  deferred_module_unloader() : thread_([this] { run(); }) {}

  ~deferred_module_unloader() {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }

  void enqueue(CUcontext ctx, CUmodule module) {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      queue_.push({ctx, module});
    }
    cv_.notify_one();
  }
};

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

  // Start the background module-cleanup thread after all devices are ready.
  module_cleanup_ = std::make_unique<deferred_module_unloader>();
}


platform::~platform() {
  // module_cleanup_ unique_ptr destructs here, joining the background thread
  // and draining all pending cuModuleUnload calls before devices/contexts are
  // torn down.  The member ordering in platform.hpp ensures this happens first.
  module_cleanup_.reset();
}

void platform::defer_module_unload(CUcontext ctx, CUmodule module) {
  if (module_cleanup_) {
    module_cleanup_->enqueue(ctx, module);
  } else {
    // Fallback in case cleanup_ was not initialised (should not happen).
    cuModuleUnload(module);
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

