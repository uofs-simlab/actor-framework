#include "caf/cuda/control-layer/pressure_scheduler.hpp"
#include "caf/cuda/control-layer/all-control-layer.hpp"
#include <iostream>

namespace caf::cuda {

//TODO overhaul this, this sucks 


pressure_scheduler::pressure_scheduler(scheduler_actor_state& state)
    : scheduler_actor_behavior(state) {
    init_state();
}

pressure_scheduler::~pressure_scheduler() = default;

void pressure_scheduler::init_state() {
    device_ = manager::get().find_device(state_.device_number);
    heuristic.emplace(device_);

    total_SM = device_->num_sms();
    available_memory = static_cast<int>(device_->total_memory_bytes());
    num_streams = state_.num_streams;

    // thresholds (tunable)
    resource_threshold = total_SM * 16; // arbitrary guard

    // concurrency thresholds: small, medium, high
    low_concurreny_threshold = 1; // 0..1 treated as low
    high_concurrency_threshold = device_->num_sms() * 4; // e.g., 4 kernels/SM

    // initial accounting
    current_concurreny = 0;
    resource_pressure = 0;
}

void pressure_scheduler::on_enter() {
    // nothing special at enter
}

void pressure_scheduler::receive(const token_ptr& tok) {
    if (tok->getType() == LAUNCH) {
        create_new_graph(tok);
        schedule();
    } else if (tok->getType() == MEMORY) {
        // For now, treat memory transfer tokens as independent graphs
        create_new_graph(tok);
        schedule();
    } else {
        // other token types: ignore or extend later
        create_new_graph(tok);
    }
}

void pressure_scheduler::schedule() {
    // Determine inflight level using current_concurreny and thresholds
    int level;
    if (current_concurreny >= high_concurrency_threshold) {
        level = HIGH;
    } else if (current_concurreny >= low_concurreny_threshold) {
        level = MEDIUM;
    } else {
        level = LOW;
    }

    // Hard cutoff: if HIGH, do not dispatch further work
    if (level == HIGH) {
        // Intentionally idle until reclaim reduces concurrency
        return;
    }

    // For Phase 1: everything classified as memory-bound. Prefer memory dispatch.
    dispatch_prefer_memory();
}

void pressure_scheduler::process_launch_token(const token_ptr& tok, int stream_id) {
    // Use the sm_usage_heuristic to compute a conservative cost
    int cost = heuristic->getCost(tok);
    if (cost == ERROR_CODE) return;

    // Update accounting similar to multilevel behavior
    // In multilevel the available resource was decremented by cost; keep that
    available_memory = std::max(0, available_memory - 0); // placeholder: token-based memory not yet available

    // Update concurrency accounting (heuristic returns SMS used; map to concurrency units)
    int concurrency_units = std::max(1, cost);
    current_concurreny += concurrency_units;
    resource_pressure += cost;

    // Build and send launch response similar to multilevel behavior
    if (tok->getType() == LAUNCH) {
        const auto& launch = static_cast<const launch_token&>(*tok);
        auto response = make_launch_response_token(state_.self, launch, state_.device_number, stream_id, cost);
        anon_mail(response).send(launch.getReplyActor());
    } else {
        // For non-launch tokens we simply log for now and assume a single unit of work
        std::cerr << "[pressure_scheduler] dispatched non-launch token on stream " << stream_id << "
";
    }
}

void pressure_scheduler::reclaim(int blocks_consumed, int memory_returned, int /*time*/, int dependency_number) {
    // Update available resources
    // blocks_consumed represents how many concurrency units to free
    current_concurreny = std::max(0, current_concurreny - blocks_consumed);
    resource_pressure = std::max(0, resource_pressure - blocks_consumed);
    available_memory = std::min(available_memory + memory_returned, static_cast<int>(device_->total_memory_bytes()));

    // If dependent graph may now be ready, re-enqueue
    if (dependency_number != INDEPENDENT) {
        if (graphs.contains(dependency_number)) {
            graph_ref ref{graph_ref::kind_t::dependent, dependency_number, 0};
            enqueue_graph_by_cost(ref);
        }
    } else {
        // Re-enqueue all independent graphs that are non-empty (best-effort)
        for (std::size_t i = 0; i < independent_graphs.size(); ++i) {
            if (!independent_graphs[i].empty()) {
                graph_ref ref{graph_ref::kind_t::independent, -1, static_cast<std::size_t>(i)};
                enqueue_graph_by_cost(ref);
            }
        }
    }

    // Attempt to schedule after resources freed
    schedule();
}

void pressure_scheduler::enqueue_graph_by_cost(const graph_ref& ref) {
    kernel_graph* g = resolve(ref);
    if (!g || g->empty()) return;

    token_ptr tok = g->peek();
    if (!tok) return;

    // For now, assume only launch tokens are costed; non-launch tokens are light
    int cost = ERROR_CODE;
    if (tok->getType() == LAUNCH) {
        cost = heuristic->getCost(tok);
        if (cost == ERROR_CODE) return;
    } else {
        cost = 1; // small cost for memory/other tokens
    }

    // Place into memory-bound queues (phase 1: everything is memory-bound)
    const long long medium_threshold = 16LL * static_cast<long long>(total_SM);
    if (cost <= total_SM) {
        low_memory_queue.push_back(ref);
    } else if (cost <= medium_threshold) {
        med_memory_queue.push_back(ref);
    } else {
        high_memory_queue.push_back(ref);
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

        int cost = ERROR_CODE;
        if (tok->getType() == LAUNCH) {
            cost = heuristic->getCost(tok);
            if (cost == ERROR_CODE) {
                q.pop_front();
                continue;
            }
        } else {
            cost = 1;
        }

        // If launching this would exceed the resource_threshold, stop for this queue
        if (resource_pressure + cost > resource_threshold) {
            break;
        }

        // If adding concurrency would exceed high_concurrency_threshold, stop
        if (current_concurreny + cost > high_concurrency_threshold) {
            break;
        }

        // Otherwise dispatch
        q.pop_front();
        token_ptr op = g->getOperation();
        if (!op) continue;

        int stream = get_next_stream();
        process_launch_token(op, stream);

        // If graph still has ops, re-enqueue for future
        if (!g->empty()) {
            enqueue_graph_by_cost(ref);
        } else {
            // If dependent, we leave removal to reclaim/gc logic elsewhere
        }
    }
}

void pressure_scheduler::dispatch_prefer_compute() {
    // Not used in Phase 1, but keep implementation symmetric
    try_dispatch_queue(low_compute_queue);
    try_dispatch_queue(med_compute_queue);
    try_dispatch_queue(high_compute_queue);
}

void pressure_scheduler::dispatch_prefer_memory() {
    // Prefer low -> med -> high memory queues
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
    int s = current_stream++ % std::max(1, num_streams);
    return s;
}

int pressure_scheduler::get_resource_pressure(int blocks_consumed) {
    // Phase 1: simple unit cost model; integrate sm_usage_heuristic later if needed
    return blocks_consumed > 0 ? blocks_consumed : 1;
}

int pressure_scheduler::get_concurrency_pressure(int blocks_consumed) {
    return get_resource_pressure(blocks_consumed);
}

} // namespace caf::cuda

