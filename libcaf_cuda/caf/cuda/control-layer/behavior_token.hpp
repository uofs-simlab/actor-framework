#include "caf/cuda/control-layer/token.hpp"
#include <string>


namespace caf::cuda {

	class behavior_token : token {
	
		public:
			behavior_token(String behavior) :
				behavior_(behavior) {}

			int getType() override {return BEHAVIOR;}
			String getBehavior() {return behavior_;}
		private:
			String behavior_;


	};



}//namespace caf::cuda
