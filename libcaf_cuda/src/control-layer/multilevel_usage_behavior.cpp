#include "caf/cuda/control-layer/multilevel_usage_behavior.hpp"
#include "caf/cuda/control-layer/all-control-layer.hpp"

namespace caf::cuda {


multilevel_usage_behavior::multilevel_usage_behavior(scheduler_actor_state& state)
    : scheduler_actor_behavior(state) {
    init_state();
}

multilevel_usage_behavior::~multilevel_usage_behavior() {}

void multilevel_usage_behavior::init_state() {
    device_ = manager::get().find_device(state_.device_number);
    heuristic.emplace(device_);
    total_SM = device_->num_sms() * 16;
    available_SM = total_SM;
    available_memory = static_cast<int>(device_->total_memory_bytes());
    num_streams = state_.num_streams;
    low_threshold = total_SM / 6;
    transfer_threshold = total_SM / 2;
    num_devices = manager::get().get_num_devices();
}

void multilevel_usage_behavior::on_enter() {

	//std::cout << "scheduler actor with device number " << state_.device_number << " Says hello\n";
	//trigger load balancing mechanisms
	if (state_.multiple_gpus) {	
		send_timed_msg();
	}

}

void multilevel_usage_behavior::send_timed_msg() {


	anon_mail(ack(TIMER)).delay(std::chrono::seconds(2)).send(state_.self);


}

void multilevel_usage_behavior::process_launch_token(const token_ptr& tok, int stream_id) {
    //scoped_timer timer("multilevel_usage_behavior::process_launch_token");

    int cost = heuristic->getCost(tok);

    const auto& launch = static_cast<const launch_token&>(*tok);
    auto response = make_launch_response_token(state_.self, launch, state_.device_number, stream_id, cost);
    anon_mail(response).send(launch.getReplyActor());
    available_SM -= cost;
}

void multilevel_usage_behavior::receive(const token_ptr& tok) {
   // scoped_timer timer("multilevel_usage_behavior::receive");

    if (tok->getType() == LAUNCH) {
        create_new_graph(tok);
	schedule();
    } else if (tok->getType() == MEMORY) {
        process_memory_transfer_token(tok, 0);
    }
}

int multilevel_usage_behavior::get_next_stream() {
    return current_stream++ % num_streams;
}

void multilevel_usage_behavior::create_new_graph(const token_ptr& tok) {
    // create graph similar to core_usage_behavior but also enqueue by cost
    if (tok->isIndependent()) {
        kernel_graph new_graph(state_.device_number, get_next_stream());
        new_graph.add_operation(tok);
        independent_graphs.push_back(std::move(new_graph));
        // reference to the newly added independent graph
        graph_ref ref{graph_ref::kind_t::independent, -1, independent_graphs.size() - 1};
        enqueue_graph_by_cost(ref);
        return;
    }

    int dep = tok->getDependency();
    
    //check to ensure that the token is not found on another device
    if (state_.multiple_gpus) {
	    int dev_num = get_device_for_dependency(dep);

	    //found elsewhere
	    if (dev_num != state_.device_number && dev_num != -1) {
	    
		    anon_mail(tok).send(state_.schedulers[dev_num]);
		    return;
	    }
    }
    
    
    
    if (graphs.contains(dep)) {
        graphs[dep].add_operation(tok);
        // If graph already existed, ensure it's enqueued only if not currently in any queue
        // For simplicity we enqueue it — caller reclaim/schedule will ensure duplicates don't cause re-dispatch
        //this may lead in an error where dependencies are triggered before they are ready
	//however since each graph gets a designated stream for now 
	//this will not until that happens 
	graph_ref ref{graph_ref::kind_t::dependent, dep};
        enqueue_graph_by_cost(ref);
    } else {
        kernel_graph new_graph(state_.device_number, get_next_stream());
        new_graph.add_operation(tok);
        graphs[dep] = std::move(new_graph);
        graph_ref ref{graph_ref::kind_t::dependent, dep};
        enqueue_graph_by_cost(ref);
    }
}

void multilevel_usage_behavior::enqueue_graph_by_cost(const graph_ref& ref) {
    kernel_graph* g = resolve(ref);
    if (!g || g->empty()) return;
    token_ptr tok = g->peek();
    if (!tok || tok->getType() != LAUNCH) return;

    int cost = heuristic->getCost(tok);
    if (cost == ERROR_CODE) return;

    // classification thresholds
    const long long medium_threshold = 16LL * static_cast<long long>(total_SM);

    if (cost <= total_SM) {
        low_queue.push_back(ref);
    } else if (cost <= medium_threshold) {
        med_queue.push_back(ref);
    } else {
        high_queue.push_back(ref);
    }
}

void multilevel_usage_behavior::try_dispatch_queue(std::deque<graph_ref>& q) {
    // Dispatch front-first while resources allow
    while (!q.empty()) {
        graph_ref ref = q.front();
        kernel_graph* g = resolve(ref);
        if (!g || g->empty()) {
            q.pop_front();
            continue;
        }

        token_ptr tok = g->peek();
        if (!tok || tok->getType() != LAUNCH) {
            // not a launch op at front => remove and continue
            q.pop_front();
            continue;
        }

        int cost = heuristic->getCost(tok);
        if (cost == ERROR_CODE) {
            q.pop_front();
            continue;
        }

        if (available_SM >= cost) {
            // we have enough resources, dispatch
            q.pop_front();
            tok = g->getOperation();
            process_launch_token(tok, g->stream_id());

            // do not reinsert; when that graph becomes ready again it will be
            // re-enqueued by reclaim(...)
        } else {
            // not enough resources for this graph; stop trying this queue
            break;
        }
    }
}

void multilevel_usage_behavior::schedule() {
   // scoped_timer timer("multilevel_usage_behavior::schedule");
    
    // Prioritize low, then medium, then high
    try_dispatch_queue(high_queue);
    try_dispatch_queue(med_queue);
    try_dispatch_queue(low_queue);
}

void multilevel_usage_behavior::reclaim(int blocks_consumed,
                                       int memory_returned,
                                       [[maybe_unused]] int time,
                                       int dependency_number) {
    // update available resources
    available_SM += blocks_consumed;
    available_memory += memory_returned;

    // If this reclaim call references a dependent graph, re-enqueue its graph
    if (dependency_number != INDEPENDENT) {
        if (graphs.contains(dependency_number)) {
            graph_ref ref{graph_ref::kind_t::dependent, dependency_number};
            enqueue_graph_by_cost(ref);
        }
    } else {
        // dependency_number < 0: we don't have a direct index for independent graphs here.
        // As a best-effort, enqueue any independent graphs that are non-empty
        //for (std::size_t i = 0; i < independent_graphs.size(); ++i) {
          ///  if (!independent_graphs[i].empty()) {
             //   graph_ref ref{graph_ref::kind_t::independent, -1, i};
               // enqueue_graph_by_cost(ref);
            //}
       // }
    }

    // After re-enqueue, attempt to schedule immediately
    schedule();
}

void multilevel_usage_behavior::reclaim(ack& return_msg) {

	//TODO IMPLEMENT TIMER ACK AND TRANSFER ACK 

	//std::cout << "scheduler actor with device number " << state_.device_number << " got an ack\n";
	if (return_msg.getType() == TIMER) {

	//std::cout << "scheduler actor with device number " << state_.device_number << " got a timer ack\n";
	
		request_load_balance();	
		send_timed_msg();	
	}

	else if (return_msg.getType() == CAF_CUDA_ACK_TRANSFER) {
		process_transfer_ack(return_msg);
	}

}

void multilevel_usage_behavior::process_transfer_ack(ack& msg) {
    // We already verified type before calling this
    auto& transfer = static_cast<transfer_ack&>(msg);

    int dep = transfer.dependency();

    // Check if we still own this dependency
    auto it = graphs.find(dep);
    if (it == graphs.end())
        return;

    graph_ref ref;
    ref.kind = graph_ref::kind_t::dependent;
    ref.dependency = dep;

    enqueue_graph_by_cost(ref);

    schedule();
}


void multilevel_usage_behavior::request_load_balance() {
	//std::cout << "Scheduler with device number " << state_.device_number << "is requesting load balance\n";  
  if (!state_.multiple_gpus)  {

        return;
  }
    // Only request work if we're underutilized

    int busy_SM = total_SM - available_SM;
    if (busy_SM > low_threshold)
    {
	 // std::cout << "Returning from since too busy\n";
        return;
    }
    //std::cout << "Hello from request load_balance\n";
    int my_device = state_.device_number;

    for (int i = 0; i < num_devices; ++i) {
        // Skip sending to self
        if (i == my_device)
            continue;

        // Send our device number to other scheduler actors
        anon_mail(my_device).urgent().send(state_.schedulers[i]);
    }
}



kernel_graph* multilevel_usage_behavior::resolve(const graph_ref& ref) {
	switch (ref.kind) {
		case graph_ref::kind_t::dependent: {
							   auto it = graphs.find(ref.dependency);
							   if (it == graphs.end()) return nullptr;
							   return &it->second;
						   }
		case graph_ref::kind_t::independent: {
							     if (ref.index >= independent_graphs.size()) return nullptr;
							     return &independent_graphs[ref.index];
						     }
	}
	return nullptr;
}


//multi GPU load balancing methods
void multilevel_usage_behavior::handle_load_balance_request(int device_number) {
    
    // Only transfer work if we are busy

    int free_SM = available_SM;
    if (free_SM > transfer_threshold) {
        return; // GPU not busy enough, do nothing
    }

    std::vector<kernel_graph> work_to_transfer;

    // ---- Helper lambda to collect transferable graphs from a queue ----
    auto collect_graphs_from_queue = [&](std::deque<graph_ref>& q, std::size_t max_count) {
        std::size_t collected = 0;

        for (auto it = q.begin(); it != q.end() && collected < max_count;) {
            graph_ref ref = *it;
            kernel_graph* g = resolve(ref);

            if (!g || g->empty() || !g->canMove()) {
                ++it;
                continue; // skip invalid or non-movable graphs
            }

            // Move the graph into the transfer vector
            work_to_transfer.push_back(std::move(*g));
            ++collected;

            // Clean up local structures
            if (ref.kind == graph_ref::kind_t::dependent) {
                remove_dependency(ref.dependency);
                graphs.erase(ref.dependency);
		add_dependency_to_device(ref.dependency,device_number);
            } else { // independent
                if (ref.index < independent_graphs.size()) {
		    //TODO come up with a better way to clean this up
                    independent_graphs[ref.index] = kernel_graph(); // reset empty
                }
            }

            // Remove from queue
            it = q.erase(it);
        }
    };

    // ---- Step 1: transfer independent graphs first ----
    std::size_t max_independent = independent_graphs.size() / 2;
    std::size_t transferred_independent = 0;

    for (std::size_t i = 0; i < independent_graphs.size() && transferred_independent < max_independent; ++i) {
        kernel_graph& g = independent_graphs[i];
        if (g.empty() || !g.canMove()) continue;

        work_to_transfer.push_back(std::move(g));
        independent_graphs[i] = kernel_graph(); // reset
        ++transferred_independent;
    }

    // ---- Step 2: transfer from high and medium queues ----
    std::size_t mid_high = high_queue.size() / 2;
    std::size_t mid_med = med_queue.size() / 2;
    std::size_t mid_low = low_queue.size() / 2;
    collect_graphs_from_queue(high_queue, mid_high);
    collect_graphs_from_queue(med_queue, mid_med);
    collect_graphs_from_queue(low_queue, mid_low);

    // ---- Step 3: send if we have anything ----
    if (!work_to_transfer.empty()) {
        anon_mail(work_to_transfer).send(state_.schedulers[device_number]);
    }
}


void multilevel_usage_behavior::receive_work(std::vector<kernel_graph> work_graphs) {
    for (auto& g : work_graphs) {

        if (g.empty() || !g.canMove())
            continue;

        token_ptr tok = g.peek();
        if (!tok)
            continue;

        int dep = tok->getDependency();

        // ============================
        // Independent Graph
        // ============================
        if (dep == INDEPENDENT) {

            g.markMoved();

            independent_graphs.push_back(std::move(g));

            graph_ref ref;
            ref.kind = graph_ref::kind_t::independent;
            ref.index = independent_graphs.size() - 1;

            enqueue_graph_by_cost(ref);
            continue;
        }

        // ============================
        // Dependent Graph
        // ============================

        // We are taking ownership of this dependency
        add_dependency_to_device(dep, state_.device_number);

        g.markMoved();

        // Store or merge into graphs map
        auto it = graphs.find(dep);
        if (it == graphs.end()) {
            graphs.emplace(dep, std::move(g));
        } else {
            // Merge operations into existing graph
            while (!g.empty()) {
                token_ptr op = g.getOperation();
                if (!op) break;
                it->second.add_operation(op);
            }
        }

        // Send transfer token to actor so it migrates dependencies
	dispatch_transfer_token(tok,g.stream_id());
   }

    // Only independent graphs were enqueued here
    schedule();
}


void multilevel_usage_behavior::add_dependency_to_device(int dependency_number, int device_number) {
	// Always override existing value
	dependency_device_map[dependency_number] = device_number;
}

void multilevel_usage_behavior::remove_dependency(int dependency_number) {
	dependency_device_map.erase(dependency_number);
}

int multilevel_usage_behavior::get_device_for_dependency(int dependency_number) const {
	auto it = dependency_device_map.find(dependency_number);
	if (it != dependency_device_map.end()) {
		return it->second;
	}
	return -1; // not found
}

} // namespace caf::cuda
