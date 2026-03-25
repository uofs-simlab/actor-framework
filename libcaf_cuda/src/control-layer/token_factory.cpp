#include "caf/cuda/control-layer/token_factory.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include "caf/cuda/control-layer/memory_response_token.hpp"
#include "caf/cuda/control-layer/transfer_token.hpp"
#include "caf/cuda/control-layer/all-control-layer.hpp"

namespace caf::cuda {

// Use manual heap allocation with intrusive_ptr
token_ptr make_launch_token(program_ptr prog,
                            nd_range range,
                            int memory_usage,
                            std::string id,
                            actor reply_to,
                            int dependency)
{
    return token_ptr(
        new launch_token(std::move(prog),
                         std::move(range),
                         memory_usage,
                         std::move(id),
                         reply_to,
                         dependency),
        caf::add_ref);
}

response_token_ptr make_launch_response_token(actor receiver,
                                              const launch_token& orig,
                                              int device_number,
                                              int stream_id,
                                              int reclaim_value,
                                              int reclaim_runtime)
{
    return response_token_ptr(
        new launch_response_token(receiver,
                                  orig,
                                  device_number,
                                  stream_id,
                                  reclaim_value,
                                  reclaim_runtime),
        caf::add_ref);
}


behavior_token_ptr make_behavior_token(std::string name)
{
    return behavior_token_ptr(new behavior_token(std::move(name)),
                              caf::add_ref);
}


token_ptr make_memory_token(int size,
                            int direction,
                            caf::actor replyActor,
                            int dependency)
{
    return token_ptr(new memory_transfer_token(size, direction, replyActor,
                                               dependency),
                     caf::add_ref);
}

response_token_ptr make_memory_response_token(actor receiver,
                                     const memory_transfer_token& orig,
				     int device_number,
				     int stream_id) {
    return response_token_ptr(new memory_response_token(receiver, orig,
                                                        device_number,
                                                        stream_id),
                              caf::add_ref);
}

response_token_ptr make_transfer_token(caf::actor receiver,
                                       const launch_token& orig,
                                       int device_number,
                                       int stream_id)
{
    return response_token_ptr(
        new transfer_token(receiver,
                           orig,
                           device_number,
                           stream_id),
        caf::add_ref);
}




/// Factory function: create a mem_ptr<int> with fake data
 mem_ptr<int> make_mem_ptr(size_t num_elements = 16) {
    if (num_elements == 1) {
        // scalar test
        return mem_ptr<int>(new mem_ref<int>(42, /*access=*/0),
                            caf::add_ref);
    } else {
        // create a device-like mem_ref with scalar backing for simplicity
        // normally this would allocate GPU memory, here just fake values
        auto ptr = mem_ptr<int>(new mem_ref<int>(0, /*CUdeviceptr*/0,
                                                 /*access=*/0),
                                caf::add_ref);
        return ptr;
    }
}


} // namespace caf::cuda

