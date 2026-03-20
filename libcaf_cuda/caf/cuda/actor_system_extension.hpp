#pragma once

// Convenience header for using the CUDA module as a CAF actor system extension.
// Include this header and use CAF_MAIN(caf::cuda::middleman) to register
// the module, then access it via sys.cuda().

#include "caf/cuda/middleman.hpp"
