#pragma once
#include <caf/all.hpp>
#include "caf/cuda/all.hpp"
#include "caf/component-actors/mmul_actor_not_square/mmul_actor_not_square.hpp"

/*
 * rotates a list of 2d vectors accoridng to a rotation matrix
 */

namespace caf::cuda {



	struct rotation_actor_state {
		caf::actor mmul_actor;
	};

	template <class T>
	caf::behavior rotation_actor_fun(caf::stateful_actor<rotation_actor_state> * self,
			caf::cuda::program_ptr mmul_kernel) {	
		using mem_t = mem_ptr<T>;
		self ->state().mmul_actor = self -> spawn(mmul_actor_NS_fun<T>,mmul_kernel);

		return {
		
			[=](mem_t rotation_matrix, //2x2 rotation matrix
			    mem_t points,
			    int num_points,
			    int device_number,
			    int stream_id) {

				int M = 2;
				int K = 2;
				int N = num_points;
				return self->mail(rotation_matrix,
						points,
						M,
						K,
						N,
						device_number,
						stream_id).delegate(self->state().mmul_actor);
				
			}
		
		};
	}




} //namespace caf::cuda
