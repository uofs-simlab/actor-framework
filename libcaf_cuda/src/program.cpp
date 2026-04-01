#include "caf/cuda/program.hpp"

namespace caf::cuda {

program::program(std::string name, std::vector<char> binary, platform_ptr platform,
                 bool is_fatbin)
    : name_(std::move(name)), binary_(std::move(binary)),
      platform_(std::move(platform)), hashValue(hasher(name_)) {
  load_kernels(is_fatbin);
}

void program::load_kernels(bool is_fatbin) {
  // Use the platform to enumerate devices
  for (const auto& dev : platform_->devices()) {
    CUcontext ctx = dev->getContext();
    CHECK_CUDA(cuCtxPushCurrent(ctx));

    CUmodule module;
    if (is_fatbin) {
      // Load fatbinary directly
      CUresult res = cuModuleLoadFatBinary(&module, binary_.data());
      if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to load fatbinary for device " +
                                 std::to_string(dev->getId()));
      }
    } else {
      // Load PTX (driver will JIT-compile for the device)
      CHECK_CUDA(cuModuleLoadData(&module, binary_.data()));
    }

    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, name_.c_str()));

    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    // Store both the module and function handles per device.
    // The module must be kept alive (and later unloaded) to keep the function valid.
    modules_[dev->getId()] = module;
    kernels_[dev->getId()] = kernel;
  }
}

program::~program() {
  // Do NOT call cuModuleUnload here directly: on some hardware (e.g. Maxwell
  // sm_52) cuModuleUnload implicitly synchronises the context, which would
  // block the calling thread (a CAF worker) for the entire kernel duration.
  // Instead we hand each (context, module) pair off to the platform's
  // background cleanup thread, which does the sync+unload there.
  for (auto& [device_id, module] : modules_) {
    try {
      CUcontext ctx = platform_->getDevice(device_id)->getContext();
      platform_->defer_module_unload(ctx, module);
    } catch (...) {
      // If we can't find the device, fall back to a direct (possibly blocking)
      // unload as a last resort rather than leaking the module handle.
      cuModuleUnload(module);
    }
  }
}

CUfunction program::get_kernel(int device_id) {
  auto it = kernels_.find(device_id);
  if (it == kernels_.end()) {
    throw std::runtime_error("Kernel not found for device ID: " +
                             std::to_string(device_id));
  }
  return it->second;
}

} // namespace caf::cuda

