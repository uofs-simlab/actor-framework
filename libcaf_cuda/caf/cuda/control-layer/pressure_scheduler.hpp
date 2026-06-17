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


#define LOW 14
#define MEDIUM 15
#define HIGH 16


namespace caf::cuda {

// - When a graph has work dispatched it is removed from the queues and
//   not re-inserted. reclaim(...) can push graphs back into queues by
//   looking up the dependency number and re-evaluating the front op.

class pressure_scheduler : public scheduler_actor_behavior {
public:
    explicit pressure_scheduler(scheduler_actor_state& state);
    ~pressure_scheduler() override;

    void on_enter() override;
    void schedule() override;
    void receive(const token_ptr& tok) override;

    // reclaim: called when resources are returned; dependency_number
    // allows this behavior to find the graph that might now be ready
    void reclaim([[maybe_unused]] int resources_consumed, [[maybe_unused]] int memory_returned, [[maybe_unused]] int time, [[maybe_unused]] int dependency_number) override;

    std::string name() const override { return "pressure_scheduler\n"; }

protected:
    void process_launch_token(const token_ptr& tok, int stream_id) override;

private:
    device_ptr device_;
    std::optional<sm_usage_heuristic> heuristic;
    void init_state();


    // resource values of the GPU
    int total_SM = 0; //can be used for proportional resource consumption
    int available_memory = 0; // bytes
    int num_streams = 0;
    int current_stream = 0;


    //threshold values
    //should inited with int_state method
    int resource_threshold; //if exceeded do not dispatch kernel
    int resource_pressure; // tracks resources in use, if low we should dispatch heavy kernels, if high dispatch light kernels 
    
    double low_concurreny_threshold; //if we under this immediately accept any work
				  //of if multiple gpus, seek out work

    double high_concurrency_threshold; //if we are above this, enqueue any work
				    //since could flood GPU with requests

    double current_concurreny; //number to assign how much kernels on the GPU
			    //values should be in proportion to how much 
			    //resources a kernel intends to consume 

    double current_sm_pressure;
    
    int compute_bound_pressure; //determines if we should favor compute or memory bound kernels when seeking work to dispatch 


    //Methods and data structures that organize and dispatch kernels
 

    //tracking graphs
    std::unordered_map<int,kernel_graph> graphs;
    std::vector<kernel_graph> independent_graphs;

    // multilevel queues of graph_refs of 
    // graphs whose next kernel is compute bound 
    std::deque<graph_ref> low_compute_queue;
    std::deque<graph_ref> med_compute_queue;
    std::deque<graph_ref> high_compute_queue;


    // multilevel queues of graph_refs of 
    // graphs whose next kernel is memory bound 
    std::deque<graph_ref> low_memory_queue;
    std::deque<graph_ref> med_memory_queue;
    std::deque<graph_ref> high_memory_queue;

    // classify & enqueue a graph reference based on its next op cost
    void enqueue_graph_by_cost(const graph_ref& ref);

    // attempt to dispatch as many graphs from q as possible (front-first)
    void try_dispatch_queue(std::deque<graph_ref>& q);

    void dispatch_prefer_compute();
    void dispatch_prefer_memory();

    kernel_graph* resolve(const graph_ref& ref);

    void create_new_graph(const token_ptr& token);

    int get_next_stream();


    //methods that return some value of resources consumed 

    //returns integer code signalling low medium or high
    //should be used in combination of thresholds to decide how much 
    //pressure it puts on a dimension of the GPU 
    //(concurreny,memory vs compute bound, resource)
    int get_resource_pressure(int blocks_consumed);
    int get_concurrency_pressure(int);
    int get_bucket_level(double); 

    double clamp_sm_ratio(double) const;

};



} // namespace caf::cuda
