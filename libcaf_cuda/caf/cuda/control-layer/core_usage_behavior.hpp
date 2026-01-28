#pragma once
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/scheduler-functions/core_heuristic_function.hpp"
#include "caf/cuda/device.hpp"

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
   int total_SM;
   int available_SM; 
   int available_memory; //in bytes 

    void init_state();


};

} // namespace caf::cuda

