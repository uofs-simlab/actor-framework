#pragma once

#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/kernel_graph.hpp"
#include "caf/cuda/control-layer/scheduler-functions/sm_usage_heuristic.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/control-layer/scheduler-functions/profiler.hpp"

#include <deque>
#include <optional>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace caf::cuda {

// Multilevel queue scheduling behavior (low / medium / high)
// - Graphs are classified by the *next operation's* cost relative to total_SM

class multilevel_usage_behavior : public scheduler_actor_behavior {
public:
    explicit multilevel_usage_behavior(scheduler_actor_state& state);
    ~multilevel_usage_behavior() override;

    void on_enter() override;
    void schedule() override;
    void receive(const token_ptr& tok) override;

    // reclaim: called when resources are returned; dependency_number
    // allows this behavior to find the graph that might now be ready
    void reclaim(int blocks_consumed, int memory_returned, [[maybe_unused]] int time, int dependency_number) override;

    //more improved version of reclaim, meant for when we need to dispatch transfer
    //tokens
    void reclaim(ack& return_msg) override;
    void process_transfer_ack(ack& msg); 

    std::string name() const override { return "multilevel_usage\n"; }


    //multi GPU load balancing methods
    //by default this scheduler behavior will try to load balance
    //across all gpus 
    void handle_load_balance_request([[maybe_unused]] int device_number) override;
    void receive_work([[maybe_unused]] std::vector<kernel_graph> work_graphs) override;
    void request_load_balance();


protected:
    int num_devices;
    void process_launch_token(const token_ptr& tok, int stream_id) override;
       

private:
    device_ptr device_;
    std::optional<sm_usage_heuristic> heuristic; //we also clamp results since
						 //it leads to more concurrent work


    int total_SM = 0; //this is not really total_SM anymore, more like a threshold 
    int available_SM = 0;
    int available_memory = 0; // bytes
    int num_streams = 0;
    int current_stream = 0;
    int low_threshold = 0; //this is used to check if we should request more work
			   //or not

    int transfer_threshold =0; //check if we should transfer work or not
    // dependency -> device mapping
    std::unordered_map<int, int> dependency_device_map;


    //tracking graphs
    std::unordered_map<int,kernel_graph> graphs;
    std::vector<kernel_graph> independent_graphs;

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

    void send_timed_msg();

    void add_dependency_to_device(int dependency_number, int device_number);
    void remove_dependency(int dependency_number);
    int get_device_for_dependency(int dependency_number) const;


};



} // namespace caf::cuda
