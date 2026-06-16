#include <algorithm>
#include "caf/cuda/manager.hpp"
#include <stdexcept>
#include <mutex>
#include <map>
#include <random>
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/token_factory.hpp" // For make_behavior_token
#include "caf/cuda/manager_config.hpp"
#include "caf/cuda/control-layer/all-control-layer.hpp"


namespace caf::cuda {

	manager* manager::instance_ = nullptr;
	std::mutex manager::mutex_;




// --------------------------------
// Static init (no config)
// --------------------------------
void manager::init(caf::actor_system& sys) {
    std::lock_guard<std::mutex> guard(mutex_);

    if (instance_) {
        throw std::runtime_error("CUDA manager already initialized");
    }

    CHECK_CUDA(cuInit(0));

    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);

    instance_ = new manager(sys);
    caf::init_global_meta_objects<caf::id_block::cuda_control>(); // Ensure control-layer types are registered

    caf::init_global_meta_objects<caf::id_block::cuda>();
}

// --------------------------------
// Static init (with config)
// --------------------------------
void manager::init(caf::actor_system& sys, manager_config config) {
    std::lock_guard<std::mutex> guard(mutex_);

    if (instance_) {
        throw std::runtime_error("CUDA manager already initialized");
    }

    CHECK_CUDA(cuInit(0));

    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);

    instance_ = new manager(sys);

    caf::init_global_meta_objects<caf::id_block::cuda>();
    caf::init_global_meta_objects<caf::id_block::cuda_control>();

    if (config.getActorBLAS()) {
        for (auto& dev : instance_->platform_->devices())
            dev->enable_cublas();
    }

    if (config.getActorSparse()) {
        for (auto& dev : instance_->platform_->devices())
            dev->enable_cusparse();
    }

  
}

int manager::get_num_devices() {return platform_ -> get_num_devices();}
// --------------------------------
// Static get()
// --------------------------------
manager& manager::get() {
    //std::lock_guard<std::mutex> guard(mutex_);

    if (!instance_) {
        throw std::runtime_error(
            "CUDA manager used before initialization.\n"
            "Call caf::cuda::manager::init() inside CAF_MAIN."
        );
    }

    return *instance_;
}

// --------------------------------
// Static shutdown()
// --------------------------------
void manager::shutdown() {
    std::lock_guard<std::mutex> guard(mutex_);

    if (!instance_)
        return;


    if (instance_->scheduler_actors_spawned_) {

    // Send exit message to all scheduler actors
    for (const auto& actor : instance_->scheduler_actors_) {
        caf::anon_mail(caf::exit_reason::user_shutdown).send(actor);
    }
  }
    delete instance_;
    instance_ = nullptr;
}

void manager::flush_programs() {
  std::unique_lock<std::shared_mutex> lock(programs_mutex_);
  programs_.clear();
}

device_ptr manager::find_device(std::size_t) const {
  throw std::runtime_error("OpenCL support disabled: manager::find_device");
}

//finds a device given its id
device_ptr manager::find_device(int id) {

	return platform_ -> getDevice(id);

}

double manager::available_memory_mb(int id) {
  auto dev = find_device(id);
  return dev ? dev->available_memory_mb() : 0.0;
}

//creates a program ptr given a kernel and a string 
program_ptr manager::create_program(const char * kernel,
                                    const std::string& name,
                                    device_ptr device) {
  size_t h = std::hash<std::string>{}(name + kernel);
  {
    std::shared_lock<std::shared_mutex> lock(programs_mutex_);
    auto it = programs_.find(h);
    if (it != programs_.end()) {
      return it->second;
    }
  }

  CUdevice current_device = device -> getDevice();

  //the compiled program can be accessed via ptx.data() afterwards
  std::vector<char> ptx;
  if (!compile_nvrtc_program(kernel,current_device,ptx)) {
    throw std::runtime_error("Program failed to compile\n");
  }

  program_ptr prog = make_counted<program>(name, ptx);

  {
    std::unique_lock<std::shared_mutex> lock(programs_mutex_);
    programs_[h] = prog;
  }

  return prog;
}
//this actually doesnt even work do not use 
program_ptr manager::create_program_from_ptx(const std::string& filename,
                                             const char* kernel_name,
                                             [[maybe_unused]] device_ptr device) {
  size_t h = std::hash<std::string>{}(filename + kernel_name);
  {
    std::shared_lock<std::shared_mutex> lock(programs_mutex_);
    auto it = programs_.find(h);
    if (it != programs_.end())
      return it->second;
  }

  static std::mutex global_ptx_mutex_map_guard;
  static std::map<std::string, std::shared_ptr<std::mutex>> ptx_mutex_map;

  // Get per-file mutex
  std::shared_ptr<std::mutex> file_mutex;
  {
    std::lock_guard<std::mutex> lock(global_ptx_mutex_map_guard);
    auto& mtx = ptx_mutex_map[filename];
    if (!mtx)
      mtx = std::make_shared<std::mutex>();
    file_mutex = mtx;
  }

  std::vector<char> ptx;
  {
    std::lock_guard<std::mutex> guard(*file_mutex);

    std::ifstream in(filename, std::ios::binary);
    if (!in) {
      throw std::runtime_error("Failed to open PTX file: " + filename);
    }

    ptx.assign(std::istreambuf_iterator<char>(in),
               std::istreambuf_iterator<char>());
  }

   // 🔒 Guard the actual JIT as well — this is the critical part!
  std::lock_guard<std::mutex> guard(*file_mutex);
  program_ptr prog = make_counted<program>(kernel_name, ptx);

  {
    std::unique_lock<std::shared_mutex> lock(programs_mutex_);
    programs_[h] = prog;
  }

  return prog;
}


//creates a program given a path to a cubin file and the kernels name
program_ptr manager::create_program_from_cubin(const std::string& filename,
                                               const char* kernel_name,
                                               [[maybe_unused]] device_ptr device) {
  size_t h = std::hash<std::string>{}(filename + kernel_name);
  {
    std::shared_lock<std::shared_mutex> lock(programs_mutex_);
    auto it = programs_.find(h);
    if (it != programs_.end())
      return it->second;
  }

  // Open the cubin file in binary mode
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open CUBIN file: " + filename);

  // Read file contents into memory
  std::vector<char> cubin((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

   // Reuse the same constructor as PTX (program class doesn't care)
  program_ptr prog = make_counted<program>(kernel_name, std::move(cubin));

  {
    std::unique_lock<std::shared_mutex> lock(programs_mutex_);
    programs_[h] = prog;
  }

  return prog;
}


//creates a program given a path to a cubin file and the kernels name
program_ptr manager::create_program_from_cubin(const std::string& filename,
                                               const char* kernel_name) {
  size_t h = std::hash<std::string>{}(filename + kernel_name);
  {
    std::shared_lock<std::shared_mutex> lock(programs_mutex_);
    auto it = programs_.find(h);
    if (it != programs_.end())
      return it->second;
  }

  // Open the cubin file in binary mode
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open CUBIN file: " + filename);

  // Read file contents into memory
  std::vector<char> cubin((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

   // Reuse the same constructor as PTX (program class doesn't care)
  program_ptr prog = make_counted<program>(kernel_name, std::move(cubin));

  {
    std::unique_lock<std::shared_mutex> lock(programs_mutex_);
    programs_[h] = prog;
  }

  return prog;
}



//creates a program given a path to a fatbin file and the kernels name
program_ptr manager::create_program_from_fatbin(const std::string& filename,
                                               const char* kernel_name) {
  size_t h = std::hash<std::string>{}(filename + kernel_name);
  {
    std::shared_lock<std::shared_mutex> lock(programs_mutex_);
    auto it = programs_.find(h);
    if (it != programs_.end())
      return it->second;
  }

  // Open the fatbin file in binary mode
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open CUBIN file: " + filename);

  // Read file contents into memory
  std::vector<char> cubin((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

   // Reuse the same constructor as PTX (program class doesn't care)
  program_ptr prog = make_counted<program>(kernel_name, std::move(cubin),true);

  {
    std::unique_lock<std::shared_mutex> lock(programs_mutex_);
    programs_[h] = prog;
  }

  return prog;
}





// Helper: Compile CUDA source to PTX for a specific device
// Returns true on success; on failure prints log and returns false
bool manager::compile_nvrtc_program(const char* source, CUdevice device, std::vector<char>& ptx_out) {

	return caf::cuda::compile_nvrtc_program(source,device,ptx_out);
}

caf::actor manager::spawn_exit_actor(int num_actors) {

	return system_.spawn(exit_actor_fun,num_actors);

}

void manager::toggle_scheduler_actor(int num_streams, int stream_depth) {
    if (scheduler_actors_spawned_)
        return;

    const auto& devices = platform_->devices();
    bool multi_gpu = devices.size() > 1;

    for (const auto& dev : devices) {
        auto hdl = system_.spawn<scheduler_actor>(
            dev->getId(), 
            num_streams, 
            stream_depth, 
            multi_gpu
        );
        scheduler_actors_.push_back(hdl);
    }

    // Link neighbors for work-stealing
    if (multi_gpu) {
        for (const auto& actor : scheduler_actors_) {
            caf::anon_mail(scheduler_actors_).send(actor);
        }
    }

    scheduler_actors_spawned_ = true;
}

void manager::enable_blas_actors() {
    for (auto& dev : platform_->devices())
        dev->enable_cublas();
}

void manager::enable_sparse_actors() {
    for (auto& dev : platform_->devices())
        dev->enable_cusparse();
}

void manager::send_scheduler_actor_message(std::vector<token_ptr> tokens) {
    if (scheduler_actors_.empty())
        return;

    size_t num_schedulers = scheduler_actors_.size();
    size_t total_tokens = tokens.size();
    size_t base_chunk = total_tokens / num_schedulers;
    size_t remainder = total_tokens % num_schedulers;

    auto it = tokens.begin();
    for (size_t i = 0; i < num_schedulers; ++i) {
        size_t count = base_chunk + (i < remainder ? 1 : 0);
        if (count == 0) continue;

        std::vector<token_ptr> chunk;
        chunk.reserve(count);
        std::move(it, it + count, std::back_inserter(chunk));
        it += count;

        caf::anon_mail(std::move(chunk)).send(scheduler_actors_[i]);
    }
}

void manager::send_scheduler_actor_message(token_ptr token) {
    if (scheduler_actors_.empty())
        return;

    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<size_t> distribution(0, scheduler_actors_.size() - 1);
    size_t idx = distribution(generator);

    caf::anon_mail(std::move(token)).send(scheduler_actors_[idx]);
}

caf::actor manager::get_scheduler_actor() {
    if (scheduler_actors_.empty()) {
        throw std::runtime_error(
            "Scheduler actors not spawned. Call manager::toggle_scheduler_actor() first."
        );
    }
    return scheduler_actors_[0];
}

caf::actor manager::get_scheduler_actor(int device_number) {
    if (scheduler_actors_.empty()) {
        throw std::runtime_error(
            "Scheduler actors not spawned. Call manager::toggle_scheduler_actor() first."
        );
    }
    
    if (device_number >= 0 && static_cast<size_t>(device_number) < scheduler_actors_.size()) {
        return scheduler_actors_[device_number];
    }
    
    // Modulo logic to handle arbitrary device numbers as per documentation
    int idx = device_number % static_cast<int>(scheduler_actors_.size());
    return scheduler_actors_[idx];
}

void manager::send_scheduler_actor_message(const std::string& behavior_name, int device_number) {
    auto target = get_scheduler_actor(device_number);
    if (target) {
        caf::anon_mail(make_behavior_token(behavior_name)).send(target);
    }
}

} // namespace caf::cuda
