#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/control-layer/return_payloads/ack.hpp"
#include <string>
#include <iostream>

/*
 * This class is meant to handle memory managment/coordination  via s/r/r IPC
 * it has nothing to do with the mem_ptr<T> (yet), that is kernel layer
 */
namespace caf::cuda {

caf::behavior scheduler_actor(caf::stateful_actor<memory_actor_state>* self, int num_devices) {

	self -> state().num_devices = num_devices;

	//initialize data structures 

	for (int i = 0; i < num_devices; i++) {
		std::queue<memory_request_token> r;
		self->state().requests.emplace_back(std::move(r));
		self->state().devices.emplace_back(manager::get().find_device(i));
	}


	return {
		[&](const memory_request_token& token) {
			int device_number = token.getDeviceNumber();
			int free_memory =
				static_cast<int>(
						self->state().devices[device_number]
						->available_memory_bytes());
			//if we have enough memory reply immediately
			if (free_memory > token.getSize()) {
				ack msg;
				self->mail(std::move(msg))
					.send(token.getReplyActor());

			}
			else {
				self->state().requests[device_number].push(std::move(token));
				if (!self->state().pending_requests) {
					//will begin a timer to check if memory is free
					anon_mail(ack(TIMER)).delay(std::chrono::seconds(1)).send(self);	
					self -> state().pending_requests = true;
				}
				//otherwise it will awake and check at a later time
			}


		},

			//typically here to peridocially check if memory is free
			[&](ack message) {

				if (message.getType() != TIMER)
					return;

				bool still_pending = false;

				for (int i = 0; i < self->state().num_devices; i++) {

					auto& q = self->state().requests[i];

					if (q.empty())
						continue;

					int device_number = i;
					int free_memory =
						static_cast<int>(self->state().devices[device_number]
								->available_memory_bytes());

					// Traverse queue until:
					//  - empty
					//  - or first unsatisfied request
					while (!q.empty()) {

						auto token = std::move(q.front());

						if (free_memory > token.getSize()) {

							q.pop();

							ack msg;
							self->mail(std::move(msg))
								.send(token.getReplyActor());

							// optionally update free_memory if allocations are immediate
							// free_memory -= token.getSize();

						} else {
							// cannot satisfy head → stop processing this queue
							break;
						}
					}

					if (!q.empty())
						still_pending = true;
				}

				if (still_pending) {
					anon_mail(ack(TIMER))
						.delay(std::chrono::seconds(1))
						.send(self);
					self->state().pending_requests = true;
				} else {
					self->state().pending_requests = false;
				}
			} //end of lambda



	};
}

} // namespace caf::cuda
