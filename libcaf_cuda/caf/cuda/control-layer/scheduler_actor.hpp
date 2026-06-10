#pragma once
#include <caf/all.hpp>
#include "caf/cuda/control-layer/token.hpp"
#include <queue>
#include <vector>
#include <random>
#include "caf/cuda/global_export.hpp"

namespace caf::cuda {

class CAF_CUDA_EXPORT scheduler_actor : public caf::event_based_actor {
public:
    scheduler_actor(caf::actor_config& cfg, int device_number, int num_streams, int stream_depth, bool multi_gpu);
    
    caf::behavior make_behavior() override;

    // Message Handler Methods
    virtual void on_receive(const token_ptr& tok);
    virtual void on_receive_batch(std::vector<token_ptr> tokens);
    virtual void on_reclaim(int val, int mem, int time, int dep, int stream_id);
    virtual void on_set_neighbors(std::vector<caf::actor> neighbors);
    virtual std::vector<token_ptr> on_steal_request(int requesting_device);

protected:
    // Scheduling logic
    virtual void schedule_work();

    // State Attributes
    int device_number_;
    int num_streams_;
    int stream_depth_;
    bool multi_gpu_;
    
    int in_flight_ = 0;
    std::queue<token_ptr> queue_;
    std::vector<caf::actor> schedulers_;
    std::vector<caf::actor> victims_;
    std::mt19937 rng_;
    std::queue<int> available_streams_;
};

} // namespace caf::cuda
