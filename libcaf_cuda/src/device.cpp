#include "caf/cuda/device.hpp"
#include "caf/cuda/program.hpp"

namespace caf::cuda {

	int device::max_active_blocks_per_sm(const program_ptr& prog, const nd_range& range,
			size_t dynamic_smem_bytes) const {
		CUfunction kernel = prog->get_kernel(id_); // full type known here
		int block_size = static_cast<int>(range.get_num_threads());
		int active_blocks = 0;
		cuOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, block_size, dynamic_smem_bytes);
		return active_blocks;
	}



}//namespace caf cuda




