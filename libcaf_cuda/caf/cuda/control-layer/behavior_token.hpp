#include "caf/cuda/control-layer/token.hpp"
#include <string>


namespace caf::cuda {

	class behavior_token : token {
	
		public:
			behavior_token(std::string behavior) :
				behavior_(behavior) {}

			int getType() override {return BEHAVIOR;}
			String getBehavior() {return behavior_;}
		private:
			std::string behavior_;


	};



}//namespace caf::cuda
