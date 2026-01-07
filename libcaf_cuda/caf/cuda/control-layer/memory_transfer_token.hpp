#pragma once
#include "caf/cuda/control-layer/token.hpp"


/*
 * This token represents transfer of memory 
 * from either host or device 
 * all size should be in bytes
 */


//direction defines
#define H2D 1
#define D2H 2


namespace caf::cuda {

class CAF_CUDA_EXPORT memory_transfer_token : public token {

	public:
		memory_transfer_token(int size,int direction):
			size_(size),
			direction_(direction) {}

		//only here to ensure that caf can copy the object for message
		//passing do not use
		memory_transfer_token() = default;

		virtual int getType() const {return MEMORY;}
		int getSize() const {return size_;}
		int getDirection() const {return direction_;}
	private:
		int size_;
		int direction_; 


};//memory transfer token class


}//caf cuda namespace

