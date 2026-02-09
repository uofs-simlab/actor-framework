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
}

void multilevel_usage_behavior::on_enter() {
    // nothing for now
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
                                       int time,
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

void multilevel_usage_behavior::reclaim(ack return_msg) {

	//TODO IMPLEMENT TIMER ACK AND TRANSFER ACK 

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
	    //TODO IMPLEMENT

    }
    void multilevel_usage_behavior::receive_work(std::vector<kernel_graph> work_graphs) {  
	    //TODO IMPLEMENT
    
    }



} // namespace caf::cuda

