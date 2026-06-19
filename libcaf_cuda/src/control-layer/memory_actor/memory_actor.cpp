#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/control-layer/memory_actor/memory_request_token.hpp"
#include "caf/cuda/control-layer/memory_actor/mem_token.hpp"
#include <iostream>

namespace caf::cuda {

caf::behavior memory_actor(caf::stateful_actor<memory_actor_state>* self,
                           int num_devices) {

    // initialize
    self->state().num_devices = num_devices;
    self->state().requests.clear();
    self->state().devices.clear();
    self->state().available_memory.clear();
    self->state().requests.resize(num_devices);
    self->state().devices.resize(num_devices);
    self->state().available_memory.resize(num_devices);

    for (int i = 0; i < num_devices; ++i) {
        auto dev = manager::get().find_device(i);
        self->state().devices[i] = dev;
        // initialize internal counter to what device reports at startup
        self->state().available_memory[i] =
            dev ? dev->available_memory_bytes() : 0;
    }

    // Handlers:
    return {
        // 1) memory request handler
        [self](const memory_request_token& token) {
            int device_number = token.getDeviceNumber();
            std::size_t req_bytes = token.getSize();

            // basic sanity checks
            if (device_number < 0 || device_number >= self->state().num_devices) {
                std::cerr << "[memory_actor] invalid device_number " << device_number << "\n";
                return;
            }

            auto &avail = self->state().available_memory[device_number];

            if (avail >= req_bytes) {
                // grant immediately: reserve in internal counter and notify requester
                avail -= req_bytes;
                // keep behavior: send ack to the reply actor so it can construct mem_token
                self->mail(std::move(mem_token(req_bytes,device_number,self))).send(token.getReplyActor());
            } else {
                // queue request (no polling). It will be reconsidered when a release arrives.
                self->state().requests[device_number].push(std::move(token));
            }
        },

        // 2) release handler: anonymous message (int device, std::size_t bytes)
        //    This is the message produced by mem_token's destructor via anon_send.
        [self](int device_number, std::size_t bytes_released) {
            if (device_number < 0 || device_number >= self->state().num_devices) {
                std::cerr << "[memory_actor] release: invalid device_number " << device_number << "\n";
                return;
            }

            // add released memory back to internal counter
            self->state().available_memory[device_number] += bytes_released;

            // try to satisfy queued requests for this device
            auto &q = self->state().requests[device_number];
            while (!q.empty()) {
                auto &front = q.front();
                std::size_t need = front.getSize();

                if (self->state().available_memory[device_number] >= need) {
                    // reserve and grant
                    self->state().available_memory[device_number] -= need;
                    self->mail(std::move(mem_token(need,device_number,self))).send(front.getReplyActor());
                    q.pop();
                } else {
                    // still can't satisfy head request
                    break;
                }
            }
        }
    };
}

} // namespace caf::cuda // namespace caf::cuda
