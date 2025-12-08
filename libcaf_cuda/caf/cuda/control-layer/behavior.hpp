#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"



//this class is meant to provide an interface so that 
//the actor can change behavior at runtime
//normally I would say use become 
//however that is too much boiler plate 
//so instead we can just build a behavior
//abstract class
//and create a dispatch table

namespace caf::cuda {
class scheduler_actor_behavior {

public:

	virtual void schedule() = 0;
	virtual void receive(launch_token token) = 0;

};


} //namespace caf::cuda

