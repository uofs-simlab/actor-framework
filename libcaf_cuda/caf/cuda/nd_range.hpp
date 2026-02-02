#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <string>
#include <functional>

namespace caf::cuda {

using dim_vec = std::vector<size_t>;

//class that represents the grid and block dimensions of a kernel
class nd_range {
public:
  // Constructor with individual dimension arguments
  nd_range(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ)
      : gridDim{static_cast<size_t>(gridX),
                static_cast<size_t>(gridY),
                static_cast<size_t>(gridZ)},
        blockDim{static_cast<size_t>(blockX),
                 static_cast<size_t>(blockY),
                 static_cast<size_t>(blockZ)} {
		 
			 computeHash();
		 
		 }


  // Constructor from vectors
  nd_range(const dim_vec& grid, const dim_vec& block) {
    if (grid.size() != 3 || block.size() != 3) {
      throw std::invalid_argument("Grid and block dimensions must each be of size 3.");
    }
    gridDim = grid;
    blockDim = block;
  }


  //default constructor
  //TODO fix issue where actors store a non in use ndrange
  nd_range() = default;


  // Getters for grid dimensions
  size_t getGridDimX() const { return gridDim[0]; }
  size_t getGridDimY() const { return gridDim[1]; }
  size_t getGridDimZ() const { return gridDim[2]; }

  // Getters for block dimensions
  size_t getBlockDimX() const { return blockDim[0]; }
  size_t getBlockDimY() const { return blockDim[1]; }
  size_t getBlockDimZ() const { return blockDim[2]; }

  // Optional: Getters for full vectors
  const dim_vec& getGridDims() const { return gridDim; }
  const dim_vec& getBlockDims() const { return blockDim; }

  // Returns total number of threads per block
  [[nodiscard]]  size_t get_num_threads() const noexcept {
	  return blockDim[0] * blockDim[1] * blockDim[2];
  }

 //get number of blocks in total
 size_t get_num_blocks() const noexcept {
	  return gridDim[0] * gridDim[1] * gridDim[2];
 }
 // Returns the precomputed hash
  [[nodiscard]] size_t getHash() const noexcept {
    return hashValue_;
  }

  ~nd_range() {
	  //no-op
  }


  // Returns a stable string representation of grid + block dims

  [[nodiscard]] std::string to_string() const {
	  std::ostringstream oss;
	  oss << "grid("
		  << gridDim[0] << ","
		  << gridDim[1] << ","
		  << gridDim[2] << ")"
		  << "_block("
		  << blockDim[0] << ","
		  << blockDim[1] << ","
		  << blockDim[2] << ")";
	  return oss.str();
  }



private:
  // Dimensions are stored in order of x, y, z
  dim_vec gridDim{3,0};
  dim_vec blockDim{3,0};
  size_t hashValue_{0}; // store precomputed hash


   // Precompute hash from to_string
  void computeHash() {
    std::hash<std::string> hasher;
    hashValue_ = hasher(to_string());
  }




};

} // namespace caf::cuda

