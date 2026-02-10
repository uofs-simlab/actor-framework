#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/all.hpp"

namespace caf::cuda {

void scheduler_actor_behavior::process_launch_token(const token_ptr& tok, int stream_id) {
    const auto& launch = static_cast<const launch_token&>(*tok);
    auto response = make_launch_response_token(state_.self, launch, state_.device_number, stream_id);
    anon_mail(response).send(launch.getReplyActor());
}


void scheduler_actor_behavior::process_memory_transfer_token(const token_ptr& tok, int stream_id) {
    const auto& mem = static_cast<const memory_transfer_token&>(*tok);
    auto response = make_memory_response_token(state_.self, mem, state_.device_number, stream_id);
    anon_mail(response).send(mem.getReplyActor());
}

} // namespace caf::cuda
