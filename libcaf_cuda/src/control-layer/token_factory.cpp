#include "caf/cuda/control-layer/token_factory.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"

namespace caf::cuda {

token_ptr make_launch_token(program_ptr prog,
                            nd_range range,
                            int memory_usage,
                            std::string id,
                            actor reply_to)
{
    return caf::make_counted<launch_token>(std::move(prog),
                                           std::move(range),
                                           memory_usage,
                                           std::move(id),
                                           reply_to);
}

token_ptr make_launch_response_token(actor receiver,
                                     const launch_token& orig)
{
    return caf::make_counted<launch_response_token>(receiver, orig);
}

behavior_token_ptr make_behavior_token(std::string name)
{
    return caf::make_counted<behavior_token>(std::move(name));
}

} // namespace caf::cuda
