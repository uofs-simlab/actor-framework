#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/single_usage_behavior.hpp"
#include "caf/cuda/control-layer/scheduler-functions/profiler.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/device.hpp"

namespace caf::cuda {

single_usage_behavior::single_usage_behavior(scheduler_actor_state& state)
    : scheduler_actor_behavior(state) {
    init_state();
}

single_usage_behavior::~single_usage_behavior() = default;

void single_usage_behavior::init_state() {
    device_ = manager::get().find_device(state_.device_number);
    // We can still keep the heuristic if you want to log real SM usage later,
    // but we won't use it for scheduling decisions
    heuristic.emplace(device_);

    // For single-usage we treat the whole GPU as occupied or free
    gpu_available = true;           // initially free
    available_memory = static_cast<int64_t>(device_->total_memory_bytes());
}

void single_usage_behavior::on_enter() {
    schedule();  // try to launch something right away if tokens already waiting
}

void single_usage_behavior::reclaim([[maybe_unused]] int blocks_consumed,
                                    int memory_returned,
                                    [[maybe_unused]] int time,
                                    [[maybe_unused]] int dependency_number) {
   
    std::cout << "reclaiming\n";
    // GPU is now free again
    gpu_available = true;
    available_memory += memory_returned;

    // Optional: could log real usage
    // std::cout << "Reclaimed: " << blocks_consumed << " SMs, " << time << " μs\n";

    schedule();  // try to launch the next one immediately
}

void single_usage_behavior::process_launch_token(const token_ptr& tok, int stream_id) {
    //scoped_timer timer("single_usage_behavior::process_launch_token");

    // For pure single-usage mode we usually don't care about the heuristic cost
    // but we can still compute it for logging / debugging
    int reported_cost = heuristic->getCost(tok);

    const auto& launch = static_cast<const launch_token&>(*tok);
    auto response = make_launch_response_token(
        state_.self,
        launch,
        state_.device_number,
        stream_id,
        reported_cost  // report real cost even if we don't use it for decision
    );

    anon_mail(response).send(launch.getReplyActor());

    // Mark GPU as busy
    gpu_available = false;
}

void single_usage_behavior::receive(const token_ptr& tok) {

    if (tok->getType() == LAUNCH) {
        create_new_graph(tok);

        // Try to dispatch immediately if GPU is currently free
        if (gpu_available) {
            schedule();
        }
    }
    else if (tok->getType() == MEMORY) {
        // For pure kernel serialisation testing you can often just forward memory ops
        // without blocking — or implement strict ordering if needed
        process_memory_transfer_token(tok, 0);
    }
}

void single_usage_behavior::create_new_graph(const token_ptr& tok) {
    if (tok->isIndependent()) {
        kernel_graph g(state_.device_number, get_next_stream());
        g.add_operation(tok);
        independent_graphs.push_back(std::move(g));
    }
    else {
        int dep = tok->getDependency();
        if (!graphs.contains(dep)) {
            kernel_graph g(state_.device_number, get_next_stream());
            graphs[dep] = std::move(g);
        }
        graphs[dep].add_operation(tok);
    }
}

int single_usage_behavior::get_next_stream() {
    return current_stream++ % num_streams;
}

// ────────────────────────────────────────────────
// The only real scheduling logic — find and launch ONE kernel if possible
// ────────────────────────────────────────────────
void single_usage_behavior::schedule() {
    if (!gpu_available) {
        return;  // GPU still busy → do nothing
    }

    token_ptr next = nullptr;
    int stream_id = -1;

    // 1. Prefer independent kernels (they have no dependencies → lowest risk)
    for (auto it = independent_graphs.begin(); it != independent_graphs.end(); ++it) {
        if (!it->empty()) {
            next = it->getOperation();
            stream_id = it->stream_id();
            if (next && next->getType() == LAUNCH) {
                // found one → launch and remove empty graph if needed
                if (it->empty()) {
                    independent_graphs.erase(it);
                }
                goto launch;
            }
        }
    }

    // 2. Otherwise take the first non-empty dependent graph (FIFO-ish)
    for (auto& [dep, graph] : graphs) {
        if (!graph.empty()) {
            next = graph.getOperation();
            stream_id = graph.stream_id();
            if (next && next->getType() == LAUNCH) {
                goto launch;
            }
        }
    }

    return;  // nothing ready to run

launch:
    process_launch_token(next, stream_id);
}

} // namespace caf::cuda
