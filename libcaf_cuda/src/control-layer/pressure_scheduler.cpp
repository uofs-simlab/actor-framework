#include "caf/cuda/control-layer/pressure_scheduler.hpp"
#include "caf/cuda/control-layer/all-control-layer.hpp"
#include <iostream>

namespace caf::cuda {

pressure_scheduler::pressure_scheduler(scheduler_actor_state& state)
    : scheduler_actor_behavior(state) {
    init_state();
}

pressure_scheduler::~pressure_scheduler() = default;

void pressure_scheduler::init_state() {
    device_ = manager::get().find_device(state_.device_number);
    heuristic.emplace(device_, false); // turn off ceiling functionality

    total_SM = device_->num_sms();
    available_memory = static_cast<int>(device_->total_memory_bytes());
    num_streams = state_.num_streams;

    // thresholds (tunable)
    resource_threshold = 1.0; // normalized SM pressure threshold (1.0 = fully loaded)

    low_concurreny_threshold = 0.25;  // low SM pressure fraction
    high_concurrency_threshold = 0.75; // high SM pressure fraction

    // initial accounting
    current_sm_pressure = 0.0;
}

void pressure_scheduler::on_enter() {
    // nothing special at enter
}

void pressure_scheduler::receive(const token_ptr& tok) {
    if (tok->getType() == LAUNCH) {
        create_new_graph(tok);
        schedule();
    } else if (tok->getType() == MEMORY) {
        process_memory_transfer_token(tok, 0);
    } else {
        create_new_graph(tok);
    }
}

int pressure_scheduler::get_bucket_level(double sm_ratio) {
    if (sm_ratio >= 0.75) return HIGH;    // big kernel
    if (sm_ratio >= 0.40) return MEDIUM;  // mid-sized
    return LOW;                           // narrow
}

void pressure_scheduler::schedule() {
    // Hard cutoff: if total SM pressure is above HIGH threshold, do not dispatch further work
    if (current_sm_pressure >= high_concurrency_threshold) {
        return;
    }

    // Phase 1: everything is memory-bound; prioritize memory queues
    dispatch_prefer_memory();
}

void pressure_scheduler::process_launch_token(const token_ptr& tok, int stream_id) {
    int sm_used = heuristic->getCost(tok);
    if (sm_used == ERROR_CODE) return;

    double sm_ratio = static_cast<double>(sm_used) / total_SM;

    // Update SM pressure accounting
    current_sm_pressure += sm_ratio;

    // Send launch response if needed
    if (tok->getType() == LAUNCH) {
        const auto& launch = static_cast<const launch_token&>(*tok);
        auto response = make_launch_response_token(state_.self, launch, state_.device_number, stream_id, sm_used);
        anon_mail(response).send(launch.getReplyActor());
    } else {
        std::cerr << "[pressure_scheduler] dispatched non-launch token on stream " << stream_id << "\n";
    }
}

void pressure_scheduler::reclaim(int sm_used, int memory_returned, int /*time*/, int dependency_number) {
    double sm_ratio = static_cast<double>(sm_used) / total_SM;
    current_sm_pressure = std::max(0.0, current_sm_pressure - sm_ratio);
    available_memory = std::min(available_memory + memory_returned, static_cast<int>(device_->total_memory_bytes()));

    if (dependency_number != INDEPENDENT) {
        if (graphs.contains(dependency_number)) {
            graph_ref ref{graph_ref::kind_t::dependent, dependency_number, 0};
            enqueue_graph_by_cost(ref);
        }
    }
    //do nothing if it was INDEPEDENDATN FOR NOW
    //WILL NEED TO CLEANUP LATER


    schedule();
}

void pressure_scheduler::enqueue_graph_by_cost(const graph_ref& ref) {
    kernel_graph* g = resolve(ref);
    if (!g || g->empty()) return;

    token_ptr tok = g->peek();
    if (!tok) return;

    int sm_used = 1; // default small cost
    if (tok->getType() == LAUNCH) {
        sm_used = heuristic->getCost(tok);
        if (sm_used == ERROR_CODE) return;
    }

    double sm_ratio = static_cast<double>(sm_used) / total_SM;
    int level = get_bucket_level(sm_ratio);

    // For now, everything is memory-bound; place in appropriate memory queue
    switch(level) {
        case LOW:    low_memory_queue.push_back(ref); break;
        case MEDIUM: med_memory_queue.push_back(ref); break;
        case HIGH:   high_memory_queue.push_back(ref); break;
    }
}

void pressure_scheduler::try_dispatch_queue(std::deque<graph_ref>& q) {
    while (!q.empty()) {
        graph_ref ref = q.front();
        kernel_graph* g = resolve(ref);
        if (!g || g->empty()) {
            q.pop_front();
            continue;
        }

        token_ptr tok = g->peek();
        if (!tok) {
            q.pop_front();
            continue;
        }

        int sm_used = (tok->getType() == LAUNCH) ? heuristic->getCost(tok) : 1;
        if (sm_used == ERROR_CODE) { q.pop_front(); continue; }

        double sm_ratio = static_cast<double>(sm_used) / total_SM;

        if (current_sm_pressure + sm_ratio > high_concurrency_threshold) break;

        q.pop_front();
        token_ptr op = g->getOperation();
        if (!op) continue;

        int stream = get_next_stream();
        process_launch_token(op, stream);

        if (!g->empty()) {
            enqueue_graph_by_cost(ref);
        }
    }
}

void pressure_scheduler::dispatch_prefer_compute() {
    try_dispatch_queue(low_compute_queue);
    try_dispatch_queue(med_compute_queue);
    try_dispatch_queue(high_compute_queue);
}

void pressure_scheduler::dispatch_prefer_memory() {
    try_dispatch_queue(low_memory_queue);
    try_dispatch_queue(med_memory_queue);
    try_dispatch_queue(high_memory_queue);
}

kernel_graph* pressure_scheduler::resolve(const graph_ref& ref) {
    if (ref.kind == graph_ref::kind_t::independent) {
        if (ref.index < independent_graphs.size()) return &independent_graphs[ref.index];
        return nullptr;
    }
    auto it = graphs.find(ref.dependency);
    if (it == graphs.end()) return nullptr;
    return &it->second;
}

void pressure_scheduler::create_new_graph(const token_ptr& token) {
    if (token->isIndependent()) {
        kernel_graph g(state_.device_number, get_next_stream());
        g.add_operation(token);
        independent_graphs.push_back(std::move(g));
        graph_ref ref{graph_ref::kind_t::independent, -1, independent_graphs.size() - 1};
        enqueue_graph_by_cost(ref);
        return;
    }

    int dep = token->getDependency();
    if (graphs.contains(dep)) {
        graphs[dep].add_operation(token);
        graph_ref ref{graph_ref::kind_t::dependent, dep, 0};
        enqueue_graph_by_cost(ref);
    } else {
        kernel_graph new_graph(state_.device_number, get_next_stream());
        new_graph.add_operation(token);
        graphs[dep] = std::move(new_graph);
        graph_ref ref{graph_ref::kind_t::dependent, dep, 0};
        enqueue_graph_by_cost(ref);
    }
}

int pressure_scheduler::get_next_stream() {
    return current_stream++ % std::max(1, num_streams);
}

int pressure_scheduler::get_resource_pressure(int blocks_consumed) {
    return blocks_consumed > 0 ? blocks_consumed : 1;
}

int pressure_scheduler::get_concurrency_pressure(int sm_used) {
    double ratio = double(sm_used) / double(total_SM);

    if (ratio < 0.10) return 1;
    if (ratio < 0.25) return 2;
    if (ratio < 0.50) return 4;
    if (ratio < 0.75) return 8;
    return 16;
}

} // namespace caf::cuda

