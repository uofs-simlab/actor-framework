#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/core_usage_behavior.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/device.hpp"

namespace caf::cuda {

core_usage_behavior::core_usage__behavior(scheduler_actor_state& state)
    : scheduler_actor_behavior(state) {
	    init_state();
    }



void core_usage_behavior::schedule() {
    //TODO IMPLEMENT
}

void core_usage_behavior::receive(const token_ptr& tok) {
    if (tok->getType() == LAUNCH) {
        //use 0 as stream id for now, eventually will have to figure out
        //stream load balancing
        process_launch_token(tok, 0);
    }
    else if (tok->getType() == MEMORY) {
        //use 0 as stream id for now, eventually will have to figure out
        //stream load balancing
        process_memory_transfer_token(tok, 0);
    }
    
}


void core_usage_behavior::init_state() {

	dev = manager::get.find_device(state-> device_number);
	heuristic = core_heuristic_function(dev);
	total_SM =  dev -> num_sms();
	available_SM = total_sm;
	available_memory = static_cast<int>(dev -> total_memory_bytes());

}

} // namespace caf::cuda
