#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/core_usage_behavior.hpp"
#include "caf/cuda/control-layer/scheduler-functions/profiler.hpp"
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
    total_SM = device_->num_sms() * 16;
    available_SM = total_SM;
    available_memory = static_cast<int>(device_->total_memory_bytes());
    num_streams = state_.num_streams;
}

void core_usage_behavior::on_enter() {
    // TODO implement
	//std::cout << "Hello\n";
}

void core_usage_behavior::reclaim(int blocks_consumed,
	       	int memory_returned,
	       	[[maybe_unused]] int time,
	       	[[maybe_unused]] int dependency_number) {

	//std::cout << "blocks is " <<  blocks_consumed << "\n";
	available_SM += blocks_consumed;
	available_memory+= memory_returned;
	//will eventually do something with the dependency number and stalling or maybe not
	schedule();

}




void core_usage_behavior::process_launch_token(const token_ptr& tok,int stream_id )  {

    
	scoped_timer timer("core_usage_behavior::process_launch_token");
	
	int cost = heuristic -> getCost(tok);

	const auto& launch = static_cast<const launch_token&>(*tok);
	auto response = make_launch_response_token(state_.self, launch, state_.device_number, stream_id,cost);
	anon_mail(response).send(launch.getReplyActor());
	available_SM -= cost;
}


void core_usage_behavior::receive(const token_ptr& tok) {

    scoped_timer timer("core_usage_behavior::receive");
    
	if (tok->getType() == LAUNCH) {
        create_new_graph(tok);
	
	//if we have the resources to dispatch, just do it right away 
	if (available_SM -  heuristic->getCost(tok) >= 0)
	 {
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

void core_usage_behavior::rank(std::size_t max_best=5) {
    scoped_timer timer("core_usage_behavior::rank");

    best_graphs.clear();

    struct candidate {
        int cost;
        graph_ref ref;
    };
    std::vector<candidate> candidates;

    // Dependent graphs
    for (auto& [dep, graph] : graphs) {
        if (graph.empty()) continue;

        token_ptr tok = graph.peek();
        if (!tok || tok->getType() != LAUNCH) continue;

        int cost = heuristic->getCost(tok);
        if (cost == ERROR_CODE) continue;

        candidates.push_back({cost,
            graph_ref{graph_ref::kind_t::dependent, dep}});
    }

    // Independent graphs
    for (std::size_t i = 0; i < independent_graphs.size(); ++i) {
        auto& graph = independent_graphs[i];
        if (graph.empty()) continue;

        token_ptr tok = graph.peek();
        if (!tok || tok->getType() != LAUNCH) continue;

        int cost = heuristic->getCost(tok);
        if (cost == ERROR_CODE) continue;

        candidates.push_back({cost,
            graph_ref{graph_ref::kind_t::independent, -1, i}});
    }

    if (candidates.empty()) return;

    std::sort(candidates.begin(), candidates.end(),
              [](const candidate& a, const candidate& b) {
                  return a.cost < b.cost;
              });

    const auto limit = std::min(max_best, candidates.size());
    for (std::size_t i = 0; i < limit; ++i) {
        best_graphs.push_back(candidates[i].ref);
    }
}



kernel_graph* core_usage_behavior::resolve(const graph_ref& ref) {
    //scoped_timer timer("core_usage_behavior::resolve");

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





void core_usage_behavior::schedule() {
    scoped_timer timer("core_usage_behavior::schedule");

    if (best_graphs.empty()) {
        rank(20); // profiled independently
        if (best_graphs.empty()) {
            return;
        }
    }

   for (std::size_t i = 0; i < best_graphs.size(); ) {
    kernel_graph* graph = resolve(best_graphs[i]);
    if (!graph || graph->empty()) {
        best_graphs.erase(best_graphs.begin() + i);
        continue; // stay at same index
    }

    token_ptr tok = graph->peek();
    if (!tok || tok->getType() != LAUNCH) {
        ++i;
        continue;
    }

    int cost = heuristic->getCost(tok);
    if (cost == ERROR_CODE) {
        ++i;
        continue;
    }

    if (available_SM >= cost) {
        tok = graph->getOperation();
        process_launch_token(tok, graph->stream_id());

        best_graphs.erase(best_graphs.begin() + i);
        continue; // stay at same index
    }

    ++i;

   }

}





} // namespace caf::cuda
