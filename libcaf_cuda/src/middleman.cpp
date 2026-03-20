#include "caf/cuda/middleman.hpp"

#include <stdexcept>
#include <fstream>
#include <mutex>
#include <map>

#include <caf/init_global_meta_objects.hpp>
#include <caf/detail/build_config.hpp>
#include <caf/detail/critical.hpp>
#include <cuda.h>

#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/all-control-layer.hpp"

namespace caf::cuda {

// -- static module interface --------------------------------------------------

void middleman::init_global_meta_objects() {
  caf::init_global_meta_objects<caf::id_block::cuda>();
}

void middleman::add_module_options(actor_system_config&) {
  // No CLI options for now. Future: add caf.cuda.scheduler-enabled, etc.
}

actor_system_module* middleman::make(actor_system& sys) {
  return new middleman(sys);
}

void middleman::check_abi_compatibility(version::abi_token token) {
  if (static_cast<int>(token) != CAF_VERSION_MAJOR) {
    CAF_CRITICAL("CAF ABI token mismatch");
  }
}

// -- constructors / destructors -----------------------------------------------

middleman::middleman(actor_system& sys) : system_(sys) {
  // Platform and CUDA init happen in start().
}

middleman::~middleman() {
  // Resources cleaned up in stop().
}

// -- actor_system_module interface --------------------------------------------

void middleman::init(actor_system_config&) {
  // Nothing to configure at this point.
}

void middleman::start() {
  // Initialize the CUDA driver API.
  CHECK_CUDA(cuInit(0));
  // Create the platform (enumerates devices, creates contexts).
  platform_ = platform::create();
}

void middleman::stop() {
  // Shut down scheduler actors.
  if (scheduler_on_) {
    for (auto& sa : scheduler_actors_) {
      anon_send_exit(sa, caf::exit_reason::user_shutdown);
    }
    scheduler_actors_.clear();
    scheduler_on_ = false;
  }
  // Shut down memory actor.
  if (memory_manager_on_) {
    destroy_memory_actor();
    memory_manager_on_ = false;
  }
  // Release platform resources.
  platform_.reset();
}

actor_system_module::id_t middleman::id() const {
  return actor_system_module::cuda_manager;
}

void* middleman::subtype_ptr() {
  return this;
}

// -- public API ---------------------------------------------------------------

device_ptr middleman::find_device(int dev_id) {
  return platform_->getDevice(dev_id);
}

int middleman::get_num_devices() {
  return platform_->get_num_devices();
}

program_ptr middleman::create_program(const char* kernel,
                                      const std::string& name,
                                      device_ptr device) {
  CUdevice current_device = device->getDevice();
  std::vector<char> ptx;
  if (!compile_nvrtc_program(kernel, current_device, ptx)) {
    throw std::runtime_error("Program failed to compile\n");
  }
  return make_counted<program>(name, ptx);
}

program_ptr middleman::create_program_from_file(const std::string& filename,
                                                const char* options,
                                                device_ptr dev) {
  // Read source file.
  std::ifstream in(filename);
  if (!in)
    throw std::runtime_error("Failed to open source file: " + filename);
  std::string source((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
  return create_program(source.c_str(), options ? options : "", dev);
}

program_ptr middleman::create_program_from_cubin(const std::string& filename,
                                                 const char* kernel_name,
                                                 device_ptr) {
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open CUBIN file: " + filename);
  std::vector<char> cubin((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  return make_counted<program>(kernel_name, std::move(cubin));
}

program_ptr middleman::create_program_from_cubin(const std::string& filename,
                                                 const char* kernel_name) {
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open CUBIN file: " + filename);
  std::vector<char> cubin((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  return make_counted<program>(kernel_name, std::move(cubin));
}

program_ptr middleman::create_program_from_fatbin(const std::string& filename,
                                                  const char* kernel_name) {
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open FATBIN file: " + filename);
  std::vector<char> fatbin((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
  return make_counted<program>(kernel_name, std::move(fatbin), true);
}

caf::actor middleman::get_scheduler_actor() {
  if (scheduler_actors_.empty())
    throw std::runtime_error("No scheduler actors initialized");
  return scheduler_actors_[0];
}

void middleman::send_scheduler_actor_message(token_ptr token,
                                             int device_number) {
  if (!scheduler_on_ || scheduler_actors_.empty())
    return;
  int num_devices = static_cast<int>(scheduler_actors_.size());
  int target = -1;
  if (device_number != -1) {
    if (device_number >= num_devices)
      return;
    target = device_number;
  } else {
    if (!token->isIndependent()) {
      target = token->getDependency() % num_devices;
      if (target < 0)
        target += num_devices;
    } else {
      target = rand() % num_devices;
    }
  }
  anon_mail(token).send(scheduler_actors_[target]);
}

void middleman::send_scheduler_actor_message(std::vector<token_ptr> tokens,
                                             int device_number) {
  if (!scheduler_on_ || scheduler_actors_.empty() || tokens.empty())
    return;
  int num_devices = static_cast<int>(scheduler_actors_.size());
  int target = -1;
  if (device_number != -1) {
    if (device_number >= num_devices)
      return;
    target = device_number;
  } else {
    target = rand() % num_devices;
  }
  anon_mail(std::move(tokens)).send(scheduler_actors_[target]);
}

void middleman::send_scheduler_actor_message(behavior_token_ptr token,
                                             int device_number) {
  if (!scheduler_on_ || scheduler_actors_.empty())
    return;
  int num_devices = static_cast<int>(scheduler_actors_.size());
  if (device_number < 0 || device_number >= num_devices)
    return;
  anon_mail(token).send(scheduler_actors_[device_number]);
}

void middleman::send_scheduler_actor_message(std::string behavior,
                                             int device_number) {
  auto token = caf::cuda::make_behavior_token(std::move(behavior));
  send_scheduler_actor_message(token, device_number);
}

caf::actor middleman::get_memory_actor() {
  if (!memory_actor_handle_)
    throw std::runtime_error("Memory actor not initialized");
  return memory_actor_handle_;
}

caf::actor middleman::spawn_exit_actor(int num_actors) {
  return system_.spawn(exit_actor_fun, num_actors);
}

// -- private helpers ----------------------------------------------------------

bool middleman::compile_nvrtc_program(const char* source, CUdevice device,
                                      std::vector<char>& ptx_out) {
  return caf::cuda::compile_nvrtc_program(source, device, ptx_out);
}

void middleman::init_scheduler_actors() {
  int num_devices = platform_->get_num_devices();
  bool multi_gpu = num_devices > 1;
  for (int i = 0; i < num_devices; i++) {
    scheduler_actors_.push_back(system_.spawn(scheduler_actor, i, multi_gpu));
  }
  if (num_devices > 1) {
    for (int i = 0; i < num_devices; i++) {
      anon_mail(scheduler_actors_).send(scheduler_actors_[i]);
    }
  }
  scheduler_on_ = true;
}

void middleman::init_memory_actor() {
  if (memory_actor_handle_)
    return;
  int num_devices = platform_->get_num_devices();
  memory_actor_handle_ = system_.spawn(memory_actor, num_devices);
  memory_manager_on_ = true;
}

void middleman::destroy_memory_actor() {
  if (!memory_actor_handle_)
    return;
  anon_send_exit(memory_actor_handle_, caf::exit_reason::user_shutdown);
  memory_actor_handle_ = caf::actor{};
}

} // namespace caf::cuda
