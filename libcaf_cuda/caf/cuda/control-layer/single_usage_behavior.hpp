#pragma once

#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/kernel_graph.hpp"
#include "caf/cuda/control-layer/scheduler-functions/sm_usage_heuristic.hpp"
#include "caf/cuda/device.hpp"

#include <unordered_map>
#include <vector>
#include <optional>

namespace caf::cuda {

class single_usage_behavior : public scheduler_actor_behavior {
public:
    explicit single_usage_behavior(scheduler_actor_state& state);
    ~single_usage_behavior() override;

    void on_enter() override;
    void schedule() override;
    void receive(const token_ptr& tok) override;
    void reclaim([[maybe_unused]] int blocks_consumed,
                 int memory_returned,
                 [[maybe_unused]] int time,
                 [[maybe_unused]] int dependency_number) override;

    std::string name() const override { return "single_usage"; }

protected:
    void process_launch_token(const token_ptr& tok, int stream_id) override;

private:
    void init_state();
    void create_new_graph(const token_ptr& tok);
    int get_next_stream();

    device_ptr device_;
    std::optional<sm_usage_heuristic> heuristic;  // optional — kept for logging/real cost reporting

    // State for single-kernel-at-a-time scheduling
    bool gpu_available = true;                    // true = GPU is idle and can accept a kernel

    // Queues for pending operations (still respect dependencies)
    std::unordered_map<int, kernel_graph> graphs;           // dependency → graph
    std::vector<kernel_graph> independent_graphs;           // no dependency

    // Stream management (even in serial mode, streams can be useful)
    int num_streams = 1;
    int current_stream = 0;

    int64_t available_memory = 0;
};

} // namespace caf::cuda
