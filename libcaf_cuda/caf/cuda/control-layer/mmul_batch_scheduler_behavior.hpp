// ========================= mmul_batch_scheduler_behavior.hpp =========================
#pragma once

#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/kernel_graph.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"

#include <deque>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace caf::cuda {

class mmul_batch_scheduler_behavior : public scheduler_actor_behavior {
public:
    explicit mmul_batch_scheduler_behavior(scheduler_actor_state& state);
    ~mmul_batch_scheduler_behavior() override;

    void on_enter() override;
    void schedule() override;
    void receive(const token_ptr& tok) override;

    void reclaim([[maybe_unused]] int blocks_consumed, [[maybe_unused]] int memory_returned, [[maybe_unused]] int time, [[maybe_unused]] int dependency_number) override;
    void reclaim(ack& return_msg) override;

    std::string name() const override { return "mmul_batch_scheduler"; }

protected:
    void process_launch_token(const token_ptr& tok, int stream_id, [[maybe_unused]] int assigned_queue);

private:
    enum queue_type { LOW = 0, MED = 1, HIGH = 2 };

    int small_block_threshold = 64;
    int medium_block_threshold = 1024;

    int max_concurrent_low = 8;
    int max_concurrent_med = 4;
    int max_concurrent_high = 1;

    int low_stream_begin = 0;
    int low_stream_end = 0;
    int med_stream_begin = 0;
    int med_stream_end = 0;
    int high_stream_begin = 0;
    int high_stream_end = 0;

    std::atomic<int> low_stream_counter{0};
    std::atomic<int> med_stream_counter{0};
    std::atomic<int> high_stream_counter{0};

    std::atomic<int> active_low{0};
    std::atomic<int> active_med{0};
    std::atomic<int> active_high{0};

    std::unordered_map<int, queue_type> dispatched_dependency_queue;

    std::deque<graph_ref> low_queue;
    std::deque<graph_ref> med_queue;
    std::deque<graph_ref> high_queue;

    device_ptr device_;
    int num_streams = 0;

    std::unordered_map<int,kernel_graph> graphs;
    std::vector<kernel_graph> independent_graphs;

    std::unordered_map<int,int> dependency_device_map;

    void init_state();
    void enqueue_graph_by_blocks(const graph_ref& ref, queue_type forced_qt = LOW);
    void try_dispatch_queue(std::deque<graph_ref>& q, queue_type qt);
    kernel_graph* resolve(const graph_ref& ref);

    int get_stream_for_queue(queue_type qt);
    queue_type classify_blocks(int blocks) const;
};

} // namespace caf::cuda
