#pragma once

#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/kernel_graph.hpp"
#include "caf/cuda/control-layer/scheduler-functions/sm_usage_heuristic.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/control-layer/profiler.hpp"

#include <deque>
#include <optional>
#include <algorithm>

namespace caf::cuda {

// Multilevel queue scheduling behavior (low / medium / high)
// - Graphs are classified by the *next operation's* cost relative to total_SM
// - Classification thresholds:
//     low:    cost <= total_SM
//     medium: cost <= 16 * total_SM
//     high:   cost >  16 * total_SM
// - schedule() will try to drain low first, then medium, then high.
// - When a graph has work dispatched it is removed from the queues and
//   not re-inserted. reclaim(...) can push graphs back into queues by
//   looking up the dependency number and re-evaluating the front op.

class multilevel_usage_behavior : public scheduler_actor_behavior {
public:
    explicit multilevel_usage_behavior(scheduler_actor_state& state);
    ~multilevel_usage_behavior() override;

    void on_enter() override;
    void schedule() override;
    void receive(const token_ptr& tok) override;

    // reclaim: called when resources are returned; dependency_number
    // allows this behavior to find the graph that might now be ready
    void reclaim(int blocks_consumed, int memory_returned, int time, int dependency_number) override;

    std::string name() const override { return "multilevel_usage\n"; }

protected:
    void process_launch_token(const token_ptr& tok, int stream_id) override;

private:
    device_ptr device_;
    std::optional<sm_usage_heuristic> heuristic;

    int total_SM = 0;
    int available_SM = 0;
    int available_memory = 0; // bytes
    int num_streams = 0;
    int current_stream = 0;

    // multilevel queues of graph_refs
    std::deque<graph_ref> low_queue;
    std::deque<graph_ref> med_queue;
    std::deque<graph_ref> high_queue;

    void init_state();
    void create_new_graph(const token_ptr& token);

    int get_next_stream();

    // classify & enqueue a graph reference based on its next op cost
    void enqueue_graph_by_cost(const graph_ref& ref);

    // attempt to dispatch as many graphs from q as possible (front-first)
    void try_dispatch_queue(std::deque<graph_ref>& q);

    kernel_graph* resolve(const graph_ref& ref);
};



} // namespace caf::cuda

