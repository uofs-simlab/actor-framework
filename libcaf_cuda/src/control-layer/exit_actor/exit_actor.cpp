#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/manager.hpp"

namespace caf::cuda {
caf::behavior exit_actor_fun(caf::stateful_actor<exit_actor_state>* self,int limit) {


        return {
                [=](int num_completed) {
                        self->state().completed += num_completed;

                        if (self->state().completed >= limit) {

                                caf::cuda::manager::shutdown();
                                self->quit();
                        }
                }
        };


}
} //namespace caf::cuda

