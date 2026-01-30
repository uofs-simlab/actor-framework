#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/core_usage_behavior.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/device.hpp"

namespace caf::cuda {

core_usage_behavior::core_usage_behavior(scheduler_actor_state& state)
    : scheduler_actor_behavior(state) {
    init_state();
}

core_usage_behavior::~core_usage_behavior(){

}

void core_usage_behavior::init_state() {
    device_ = manager::get().find_device(state_.device_number);
    heuristic.emplace(device_);
    total_SM = device_->num_sms();
    available_SM = total_SM;
    available_memory = static_cast<int>(device_->total_memory_bytes());
    num_streams = state_.num_streams;
}

void core_usage_behavior::on_enter() {
    // TODO implement
}

void core_usage_behavior::reclaim(int, int, int, int) {
    // intentionally empty for now
}

void core_usage_behavior::schedule() {	
	//dummy_schedule();
}


void core_usage_behavior::process_launch_token(const token_ptr& tok,int stream_id )  {

	scheduler_actor_behavior::process_launch_token(tok,stream_id);
	available_SM -= heuristic->getCost(tok);

}


void core_usage_behavior::receive(const token_ptr& tok) {
    if (tok->getType() == LAUNCH) {
        create_new_graph(tok);
	if (available_SM -  heuristic->getCost(tok) < 0) {schedule();}
	else {
		if (tok->isIndependent()) 
		{
			process_launch_token(tok,get_next_stream());
			return;
		}

		int stream = graphs[tok->getDependency()].stream_id();
		process_launch_token(tok,stream);
	}

    } else if (tok->getType() == MEMORY) {
        process_memory_transfer_token(tok, 0);
    }
}

int core_usage_behavior::get_next_stream() {
    return current_stream++ % num_streams;
}

void core_usage_behavior::create_new_graph(const token_ptr& tok) {
    if (tok->isIndependent()) {
        kernel_graph new_graph(state_.device_number, get_next_stream());
        new_graph.add_operation(tok);
        independent_graphs.push_back(std::move(new_graph));
        return;
    }
    else if (graphs.contains(tok->getDependency())) {
        graphs[tok->getDependency()].add_operation(tok);
    }
    else {
        kernel_graph new_graph(state_.device_number, get_next_stream());
        new_graph.add_operation(tok);
        graphs[tok->getDependency()] = std::move(new_graph);
    }
}

void core_usage_behavior::dummy_schedule() {
    // Drain independent graphs fully
    for (auto it = independent_graphs.begin(); it != independent_graphs.end(); ) {
        kernel_graph& graph = *it;
        while (!graph.empty()) {
            token_ptr tok = graph.getOperation();
            if (!tok) break;
            if (tok->getType() == LAUNCH)
                process_launch_token(tok, graph.stream_id());
        }
        it = independent_graphs.erase(it);
    }

    // Drain dependency graphs, do not delete
    for (auto& [dep, graph] : graphs) {
        while (!graph.empty()) {
            token_ptr tok = graph.getOperation();
            if (!tok) break;
            if (tok->getType() == LAUNCH)
                process_launch_token(tok, graph.stream_id());
        }
    }
}

void core_usage_behavior::rank(std::size_t max_best = 5) {
   
   //TODO INCORPORATE GRAPH STATUS INTO THIS EVENTUALLY
   //AND FIGURE OUT A WAY TO CUT OUT EMPTY GRAPHS	
	
   best_graphs.clear();

    struct candidate {
        int cost;
        kernel_graph* graph;
    };

    std::vector<candidate> candidates;
    candidates.reserve(graphs.size());

    // Step 1: gather all candidate graphs
    for (auto& [dep, graph] : graphs) {
        if (graph.empty())
            continue;

        token_ptr tok = graph.peek();  // non-destructive
        if (!tok || tok->getType() != LAUNCH)
            continue;

        int cost = heuristic->getCost(tok);
        if (cost == ERROR_CODE)
            continue;

        candidates.push_back({cost, &graph});
    }

    if (candidates.empty())
        return;

    // Step 2: sort by ascending cost (cheap kernels first)
    std::sort(candidates.begin(), candidates.end(),
              [](const candidate& a, const candidate& b) {
                  return a.cost < b.cost;
              });

    // Step 3: keep top N candidates
    const std::size_t limit = std::min(max_best, candidates.size());
    for (std::size_t i = 0; i < limit; ++i) {
        best_graphs.push_back(candidates[i].graph);
    }
}




} // namespace caf::cuda
