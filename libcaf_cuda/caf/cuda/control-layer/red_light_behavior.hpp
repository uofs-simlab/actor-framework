#pragma once
#include "caf/cuda/control-layer/behavior.hpp"

namespace caf::cuda {

class red_light_behavior : public scheduler_actor_behavior {
public:
    explicit red_light_behavior(scheduler_actor_state& state);
    void on_enter() override;
    void schedule() override;
    void receive(const token_ptr& tok) override;
    ~red_light_behavior() noexcept override;

protected:
    virtual void process_launch_token(const token_ptr& tok, int stream_id);
    virtual void process_memory_transfer_token(const token_ptr& tok, int stream_id);

};

} // namespace caf::cuda
