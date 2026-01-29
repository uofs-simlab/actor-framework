#pragma once
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/kernel_graph.hpp"
#include "caf/cuda/control-layer/scheduler-functions/core_heuristic_function.hpp"
#include "caf/cuda/device.hpp"
#include <unordered_map>

namespace caf::cuda {

class core_usage_behavior : public scheduler_actor_behavior {
public:
    explicit core_usage_behavior(scheduler_actor_state& state);
    void on_enter() override;
    void schedule() override;
    void receive(const token_ptr& tok) override;
    void reclaim(int value /*blocks consumed*/,int memory_returned,int time,int dependency) override; 

private:
   device_ptr device_;
   core_heuristic_function heuristic;
  
   //tracking the resources of the device 
   int total_SM;
   int available_SM; 
   int available_memory; //in bytes 
   int num_tokens = 0;
   
   int num_streams = 0;
   int current_stream = 0;


   //data structures to manage dependencies
   std::unordered_map<int,kernel_graph> graphs;
   std::vector<kernel_graph> independent_graphs;
   std::vector<kernel_graph> best_graphs; //should contain top 5-10 best selections ideally or something along the lines

   void init_state();
   void create_new_graph(token_ptr& token); //this should either add to indepedent or graphs data structure
			    //note to self use std::move for cheap copies
   
   void rank(); //this should rank the graphs (high to low) for best canidates

   int get_next_stream(); // this should return the next stream based on some decisions

};

} // namespace caf::cuda

