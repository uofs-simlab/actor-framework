#include "caf/cuda/manager.hpp"
#include <stdexcept>
#include <mutex>
#include <map>
#include "caf/cuda/control-layer/scheduler_actor.hpp"
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

    instance_->scheduler_on = config.getSchedulerOn();
    instance_->memory_manager_on = config.getMemoryManagerOn();

    if (instance_->scheduler_on) { 
	    instance_ -> init_scheduler_actors(sys);
    }
    if (instance_->memory_manager_on) {
	    instance_->init_memory_actor(sys);
    }



}

int manager::get_num_devices() {return platform_ -> get_num_devices();}

void manager::init_scheduler_actors(caf::actor_system& sys) {

	int num_devices = platform_ -> get_num_devices();

	bool multi_gpu = num_devices > 1;
	for (int i = 0; i < num_devices; i++) {
	
		instance_ -> scheduler_actors.push_back( sys.spawn(scheduler_actor,i,multi_gpu));
	
	}

	//if there is multiple GPUs send every scheduler actor contact information 
	//about the other on	
	if (num_devices > 1) {
	
		for (int i = 0; i < num_devices; i++) {
			anon_mail(scheduler_actors).send(instance_ -> scheduler_actors[i]);
		}
	
	}

}




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

    if (instance_->scheduler_on) {
        
	for (int i = 0; i < instance_ ->  platform_ -> get_num_devices(); i++) {
	    
	anon_send_exit(
            instance_->scheduler_actors[i],
            caf::exit_reason::user_shutdown
        );



    }

    }


    if (instance_->memory_manager_on) {
	    instance_->destroy_memory_actor();
    }


    delete instance_;
    instance_ = nullptr;
}

// --------------------------------
// Static getter for scheduler actor
// --------------------------------
// this is legacy code, do not use
// only exists to be backwards compatable with tests 
caf::actor manager::get_scheduler_actor() {
    	//this is a read only data no need for lock
	//std::lock_guard<std::mutex> guard(mutex_);


    if (!instance_) {
        throw std::runtime_error("CUDA manager not initialized");
    }

    return instance_->scheduler_actors[0];
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
  

	CUdevice current_device = device -> getDevice();;

	//the compiled program can be accessed via ptx.data() afterwards
	std::vector<char> ptx;
        if (!compile_nvrtc_program(kernel,current_device,ptx)) {
	
		throw std::runtime_error("Program failed to compile\n");
	
	}
	program_ptr prog = make_counted<program>(name, ptx);
	return prog;
}
//this actually doesnt even work do not use 
program_ptr manager::create_program_from_ptx(const std::string& filename,
                                             const char* kernel_name,
                                             [[maybe_unused]] device_ptr device) {
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
  return prog;
}


//creates a program given a path to a cubin file and the kernels name
program_ptr manager::create_program_from_cubin(const std::string& filename,
                                               const char* kernel_name,
                                               [[maybe_unused]] device_ptr device) {
  // Open the cubin file in binary mode
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open CUBIN file: " + filename);

  // Read file contents into memory
  std::vector<char> cubin((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

   // Reuse the same constructor as PTX (program class doesn't care)
  program_ptr prog = make_counted<program>(kernel_name, std::move(cubin));
  return prog;
}


//creates a program given a path to a cubin file and the kernels name
program_ptr manager::create_program_from_cubin(const std::string& filename,
                                               const char* kernel_name) {
  // Open the cubin file in binary mode
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open CUBIN file: " + filename);

  // Read file contents into memory
  std::vector<char> cubin((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

   // Reuse the same constructor as PTX (program class doesn't care)
  program_ptr prog = make_counted<program>(kernel_name, std::move(cubin));
  return prog;
}



//creates a program given a path to a fatbin file and the kernels name
program_ptr manager::create_program_from_fatbin(const std::string& filename,
                                               const char* kernel_name) {
  // Open the fatbin file in binary mode
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open CUBIN file: " + filename);

  // Read file contents into memory
  std::vector<char> cubin((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

   // Reuse the same constructor as PTX (program class doesn't care)
  program_ptr prog = make_counted<program>(kernel_name, std::move(cubin),true);
  return prog;
}





// Helper: Compile CUDA source to PTX for a specific device
// Returns true on success; on failure prints log and returns false
bool manager::compile_nvrtc_program(const char* source, CUdevice device, std::vector<char>& ptx_out) {

	return caf::cuda::compile_nvrtc_program(source,device,ptx_out);
}

// ---------------------------------------------
// Send single token
// ---------------------------------------------
void manager::send_scheduler_actor_message(token_ptr token, int device_number) {
    if (!scheduler_on || scheduler_actors.empty())
        return;

    int num_devices = static_cast<int>(scheduler_actors.size());
    int target = -1;

    if (device_number != -1) {
        // Explicit device
        if (device_number >= num_devices)
            return; // silently discard
        target = device_number;
    } else {
        // No device specified
        if (!token->isIndependent()) {
            target = token->getDependency() % num_devices;
            if (target < 0)
                target += num_devices;
        } else {
            target = rand() % num_devices;
        }
    }

    anon_mail(token).send(scheduler_actors[target]);
}

// ---------------------------------------------
// Send vector of tokens
// ---------------------------------------------
void manager::send_scheduler_actor_message(std::vector<token_ptr> tokens,
                                           int device_number) {
    if (!scheduler_on || scheduler_actors.empty() || tokens.empty())
        return;

    int num_devices = static_cast<int>(scheduler_actors.size());
    int target = -1;

    if (device_number != -1) {
        // Explicit device
        if (device_number >= num_devices)
            return; // silently discard
        target = device_number;
    } else {
        // No device specified → random
        target = rand() % num_devices;
    }

    anon_mail(std::move(tokens)).send(scheduler_actors[target]);
}

void manager::send_scheduler_actor_message(behavior_token_ptr token, int device_number) {
    if (!scheduler_on || scheduler_actors.empty())
        return;

    int num_devices = static_cast<int>(scheduler_actors.size());

    // Drop if device number is invalid
    if (device_number < 0 || device_number >= num_devices)
        return;

    anon_mail(token).send(scheduler_actors[device_number]);
}

void manager::send_scheduler_actor_message(std::string behavior, int device_number) {
    auto token = caf::cuda::make_behavior_token(std::move(behavior));
    send_scheduler_actor_message(token, device_number);
}


void manager::init_memory_actor(caf::actor_system& sys) {
    if (memory_actor_handle)
        return; // already initialized

    int num_devices = platform_->get_num_devices();

    memory_actor_handle =
        sys.spawn(memory_actor, num_devices);
}

void manager::destroy_memory_actor() {
    if (!memory_actor_handle)
        return;

    anon_send_exit(
        memory_actor_handle,
        caf::exit_reason::user_shutdown
    );

    memory_actor_handle = caf::actor{};
}

caf::actor manager::get_memory_actor() {
    if (!instance_ || !instance_->memory_actor_handle) {
        throw std::runtime_error("Memory actor not initialized");
    }

    return instance_->memory_actor_handle;
}

caf::actor manager::spawn_exit_actor(int num_actors) {

	return system_.spawn(exit_actor_fun,num_actors);

}


} // namespace caf::cuda
