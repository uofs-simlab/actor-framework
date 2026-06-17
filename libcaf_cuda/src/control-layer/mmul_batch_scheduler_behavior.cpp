
// ========================= mmul_batch_scheduler_behavior.cpp =========================

#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/mmul_batch_scheduler_behavior.hpp"
#include "caf/cuda/control-layer/kernel_graph.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/all-control-layer.hpp"

#include <deque>
#include <optional>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace caf::cuda {

// A scheduler tuned for batched matrix-multiply kernels (mmul).
// Classification is based on the kernel grid block count instead of SM occupancy.
// Streams are partitioned per-class and each class has a configurable maximum
// concurrent kernels.

// -------------------- implementation --------------------

mmul_batch_scheduler_behavior::mmul_batch_scheduler_behavior(scheduler_actor_state& state)
    : scheduler_actor_behavior(state) {
    init_state();
}

mmul_batch_scheduler_behavior::~mmul_batch_scheduler_behavior() {}

void mmul_batch_scheduler_behavior::init_state() {
    device_ = manager::get().find_device(state_.device_number);
    num_streams = 128;

    // default partitioning: LOW:50% of streams, MED:25%, HIGH:25%
    low_stream_begin = 0;
    low_stream_end = std::max(1, num_streams / 2);

    med_stream_begin = low_stream_end;
    med_stream_end = med_stream_begin + std::max(1, num_streams / 4);

    high_stream_begin = med_stream_end;
    high_stream_end = std::max(high_stream_begin + 1, num_streams);

    // clamp ends to num_streams
    if (low_stream_end > num_streams) low_stream_end = num_streams;
    if (med_stream_end > num_streams) med_stream_end = num_streams;
    if (high_stream_end > num_streams) high_stream_end = num_streams;

    // thresholds and max concurrents can be tuned by editing these members
    small_block_threshold = 128;
    medium_block_threshold = 512;

    max_concurrent_low = 20000;
    max_concurrent_med = 16000;
    max_concurrent_high = 60000;
}

void mmul_batch_scheduler_behavior::on_enter() {
    // nothing special for now
}

void mmul_batch_scheduler_behavior::receive(const token_ptr& tok) {
    if (!tok) return;

    if (tok->getType() == LAUNCH) {
        // Inspect block count now to choose queue + stream before creating the graph.
        const auto& launch = static_cast<const launch_token&>(*tok);
        int blocks = launch.getBlocks();
        queue_type qt = classify_blocks(blocks);
        int assigned_stream = get_stream_for_queue(qt);

        if (tok->isIndependent()) {
            // create independent graph with preselected stream
            kernel_graph new_graph(state_.device_number, assigned_stream);
            new_graph.add_operation(tok);
            independent_graphs.push_back(std::move(new_graph));
            graph_ref ref{graph_ref::kind_t::independent, -1, independent_graphs.size() - 1};
            enqueue_graph_by_blocks(ref, qt);
            schedule();
            return;
        }

        int dep = tok->getDependency();

        // If multiple GPUs are in play, forward to the owning device if found
        if (state_.multiple_gpus) {
            int dev_num = -1;
            auto it = dependency_device_map.find(dep);
            if (it != dependency_device_map.end()) dev_num = it->second;

            if (dev_num != state_.device_number && dev_num != -1) {
                // preserve stream choice on target side by forwarding token (target will reclassify)
                anon_mail(tok).send(state_.schedulers[dev_num]);
                return;
            }
        }

        if (graphs.contains(dep)) {
            // append operation to existing graph; keep the stream already assigned to that graph
            graphs[dep].add_operation(tok);
            graph_ref ref{graph_ref::kind_t::dependent, dep};
            enqueue_graph_by_blocks(ref, qt);
        } else {
            // create a new dependent graph, assign the stream we selected earlier
            kernel_graph new_graph(state_.device_number, assigned_stream);
            new_graph.add_operation(tok);
            graphs[dep] = std::move(new_graph);
            graph_ref ref{graph_ref::kind_t::dependent, dep};
            // record that we own this dependency on this device
            dependency_device_map[dep] = state_.device_number;
            enqueue_graph_by_blocks(ref, qt);
        }

        schedule();

    } else if (tok->getType() == MEMORY) {
        process_memory_transfer_token(tok, 0);
    }
}

mmul_batch_scheduler_behavior::queue_type mmul_batch_scheduler_behavior::classify_blocks(int blocks) const {
    if (blocks <= small_block_threshold) return LOW;
    if (blocks <= medium_block_threshold) return MED;
    return HIGH;
}

void mmul_batch_scheduler_behavior::enqueue_graph_by_blocks(const graph_ref& ref, queue_type forced_qt) {
    kernel_graph* g = resolve(ref);
    if (!g || g->empty()) return;

    // If caller provided forced_qt, prefer it. Otherwise classify by peeking the front op.
    queue_type qt = forced_qt;

    // push into the appropriate queue
    switch (qt) {
        case LOW:  low_queue.push_back(ref); break;
        case MED:  med_queue.push_back(ref); break;
        case HIGH: high_queue.push_back(ref); break;
    }
}

int mmul_batch_scheduler_behavior::get_stream_for_queue(queue_type qt) {
    if (qt == LOW) {
        int range = std::max(1, low_stream_end - low_stream_begin);
        int idx = low_stream_counter++ % range;
        return low_stream_begin + idx;
    }
    if (qt == MED) {
        int range = std::max(1, med_stream_end - med_stream_begin);
        int idx = med_stream_counter++ % range;
        return med_stream_begin + idx;
    }
    // HIGH
    int range = std::max(1, high_stream_end - high_stream_begin);
    int idx = high_stream_counter++ % range;
    return high_stream_begin + idx;
}

void mmul_batch_scheduler_behavior::try_dispatch_queue(std::deque<graph_ref>& q, queue_type qt) {
    while (!q.empty()) {
        // check concurrency cap for this queue
        if (qt == LOW && active_low.load() >= max_concurrent_low) break;
        if (qt == MED && active_med.load() >= max_concurrent_med) break;
        if (qt == HIGH && active_high.load() >= max_concurrent_high) break;

        graph_ref ref = q.front();
        kernel_graph* g = resolve(ref);
        if (!g || g->empty()) {
            q.pop_front();
            continue;
        }

        token_ptr tok = g->peek();
        if (!tok || tok->getType() != LAUNCH) {
            q.pop_front();
            continue;
        }

        // Use the stream pre-assigned to the graph at creation time
        int stream = g->stream_id();

        // pop from queue before launching
        q.pop_front();

        // take the operation from the graph
        token_ptr op = g->getOperation();
        if (!op) continue;

        int dep = op->getDependency();

        // increment active counter and record mapping by dependency (if dependent)
        switch (qt) {
            case LOW:  active_low.fetch_add(1); break;
            case MED:  active_med.fetch_add(1); break;
            case HIGH: active_high.fetch_add(1); break;
        }

        if (dep != INDEPENDENT) {
            dispatched_dependency_queue[dep] = qt;
        } else {
            // Independent graphs: we don't have a unique dependency id to map on reclaim.
            // Best-effort: nothing to record. If independent graphs are common, consider
            // generating a unique id per-independent-graph and setting it in the
            // launch_response_token's reclaim_dependency_ so reclaim(...) can map it back.
        }

        process_launch_token(op, stream, static_cast<int>(qt));
    }
}

void mmul_batch_scheduler_behavior::schedule() {
    // Favor small kernels first to keep latency low and allow concurrency
    try_dispatch_queue(low_queue, LOW);
    try_dispatch_queue(med_queue, MED);
    try_dispatch_queue(high_queue, HIGH);
}

void mmul_batch_scheduler_behavior::process_launch_token(const token_ptr& tok, int stream_id, [[maybe_unused]] int assigned_queue) {
    // Create a launch response token and send it (same pattern as multilevel)
    const auto& launch = static_cast<const launch_token&>(*tok);
    auto response = make_launch_response_token(state_.self, launch, state_.device_number, stream_id, /*reclaim_value*/ 0, /*reclaim_runtime*/ 0);
    anon_mail(response).send(launch.getReplyActor());

    // Note: actual reclaim will arrive via anon_mail(reclaim_value, reclaim_memory, reclaim_runtime, reclaim_dependency)
    // when launch_response_token::release() runs on the device side. That triggers reclaim(...) in this actor.
}

void mmul_batch_scheduler_behavior::reclaim([[maybe_unused]] int blocks_consumed,
                                            [[maybe_unused]] int memory_returned,
                                            [[maybe_unused]] int time,
                                            int dependency_number) {
    // This reclaim() is called when the device (or the launch_response_token destructor)
    // sends the 4-tuple (reclaim_value, reclaim_memory, reclaim_runtime, reclaim_dependency).

    // resource tracking (if other components use available_SM/available_memory)
    //available_SM += blocks_consumed;
    //available_memory += memory_returned;

    // If reclaim references a dependent graph, decrement the active counter we recorded
    if (dependency_number != INDEPENDENT) {
        auto it = dispatched_dependency_queue.find(dependency_number);
        if (it != dispatched_dependency_queue.end()) {
            queue_type qt = it->second;
            dispatched_dependency_queue.erase(it);
            switch (qt) {
                case LOW:  active_low.fetch_sub(1); break;
                case MED:  active_med.fetch_sub(1); break;
                case HIGH: active_high.fetch_sub(1); break;
            }
        }

        // re-enqueue the graph if there are queued operations on that dependency
        if (graphs.contains(dependency_number)) {
            graph_ref ref{graph_ref::kind_t::dependent, dependency_number};
            enqueue_graph_by_blocks(ref);
        }
    } else {
        // Independent graphs
        // The reclaim reports INDEPENDENT as the dependency. Without a unique ID we can't map
        // this reclaim back to a specific independent graph's active counter. If independent
        // graphs are common, add a unique graph id into the reclaim fields in
        // launch_response_token so we can correctly decrement the active counter here.
    }

    // attempt to schedule immediately after resources returned
    schedule();
}

void mmul_batch_scheduler_behavior::reclaim(ack& return_msg) {
    // The actor-level ack path: if the device or other parts send specialized acks
    // (e.g. TIMER, CAF_CUDA_ACK_TRANSFER), handle them here.
    if (return_msg.getType() == TIMER) {
        // no-op for now
        return;
    } else if (return_msg.getType() == CAF_CUDA_ACK_TRANSFER) {
        //process_transfer_ack(return_msg);
    }
}

kernel_graph* mmul_batch_scheduler_behavior::resolve(const graph_ref& ref) {
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

} // namespace caf::cuda
