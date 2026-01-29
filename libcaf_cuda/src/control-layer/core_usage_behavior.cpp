#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/core_usage_behavior.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/device.hpp"

namespace caf::cuda {

core_usage_behavior::core_usage_behavior(scheduler_actor_state& state)
    : scheduler_actor_behavior(state) {
	    init_state();
    }

void core_usage_behavior::init_state() {

	dev = manager::get().find_device(state-> device_number);
	heuristic = core_heuristic_function(device_);
	total_SM =  device_ -> num_sms();
	available_SM = total_SM;
	available_memory = static_cast<int>(device_ -> total_memory_bytes());
	num_streams = state -> num_streams;

}

void core_usage_behavior::on_enter() {
	//TODO implement 
	//

}


void core_usage_behavior::schedule() {
    //TODO IMPLEMENT

	dummy_schedule();
}

void core_usage_behavior::receive(const token_ptr& tok) {
    if (tok->getType() == LAUNCH) {
        //use 0 as stream id for now, eventually will have to figure out
        //stream load balancing
    	create_new_graph(token);
	schedule();	
    }
    else if (tok->getType() == MEMORY) {
        //use 0 as stream id for now, eventually will have to figure out
        //stream load balancing
        process_memory_transfer_token(tok, 0);
    }
    
}



int core_usage_behavior::get_next_stream() { return current_stream++ % num_streams;}

void core_usage_behavior::create_new_graph(token_ptr& token) {

	//check for independence first 
	if (token -> isIndependent()) {
		
        	kernel_graph new_graph(state -> device_number, get_next_stream());
		
		new_graph.add_operation(token);
		independent_graphs.push_back(std::move(new_graph));
		return;
	}
	//check if we have seen the operation already 
	else if (graphs.contains(token -> getDependency())) {
		graphs[token-> getDependency()].add_operation(token);
	}

	//at this point this is the first time we are seeing this add to hashmap
	else {
	
        	kernel_graph new_graph(state -> device_number, get_next_stream());	
		new_graph.add_operation(token);
		graphs[token -> getDependency()] = std::move(new_graph);

	}


}



void core_usage_behavior::dummy_schedule() {
    /*
     * 1. Drain independent graphs fully.
     *    These can be erased once empty.
     */
    for (auto it = independent_graphs.begin();
         it != independent_graphs.end(); ) {

        kernel_graph& graph = *it;

        while (!graph.empty()) {
            token_ptr tok = graph.getOperation();
            if (!tok)
                break;

            if (tok->getType() == LAUNCH) {
                process_launch_token(tok, graph.get_stream_id());
            }
            // ignore other token types for now
        }

        // independent graphs can be deleted once drained
        it = independent_graphs.erase(it);
    }

    /*
     * 2. Drain dependency graphs, but DO NOT delete them.
     */
    for (auto& [dep, graph] : graphs) {
        while (!graph.empty()) {
            token_ptr tok = graph.getOperation();
            if (!tok)
                break;

            if (tok->getType() == LAUNCH) {
                process_launch_token(tok, graph.get_stream_id());
            }
        }
    }
}




} // namespace caf::cuda
