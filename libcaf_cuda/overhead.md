Root causes identified
Two distinct sources of overhead were found by tracing through command_runner::run → command::enqueue → device::collect_output_buffers → mem_ref::copy_to_host → helpers::extract_vector:

1. extract_vector was copying the entire output buffer (primary fix — fully eliminated)
caf::cuda::extract_vector<int>(result_buffer) took const std::vector<output_buffer>&, found the inner std::vector<int> via std::get_if, and did return *ptr — a full copy of the N×N integer matrix. For N=16000 that is 1 GB being copied purely to transfer pointer ownership. This was eliminating all the benefit of the move chain inside enqueue.

Fix: Added && overloads for both extract_vector and extract_vector_or_empty in libcaf_cuda/caf/cuda/helpers.hpp that do return std::move(*ptr). Updated both main_actor_command_runner.cpp (using std::move(result_buffer)) and main_actor_facade.cpp (callback changed from const result_vec& to result_vec by value, then std::move(result)).

Result: extract_vector at N=16000 went from 832 ms → 0.0004 ms.

2. copy_to_host() allocates the D2H buffer inside the timed window (structural issue)
mem_ref::copy_to_host() does std::vector<T> host_data(num_elements_) which:

Calls malloc for N² ints (~1 GB at N=16000)
Zero-initializes all elements, which on Linux demand-paging triggers ~262K minor page faults that physically back all pages
Native CUDA pre-allocates h_c before any timer starts, so these page faults are invisible in native's measurement. In the actor code they happen inside runner.run. This is ~1200 ms overhead for N=16000.

Skipping zero-init is not a simple fix here: if pages aren't pre-backed, CUDA's pageable-memory DMA staging causes the same page faults during the actual copy. The right fix requires either: (a) pre-allocating the output buffer outside the timed window and passing it into runner.run, or (b) using cuMemAllocHost pinned memory (which changes the fair-comparison baseline). This is an API-level change to address separately.

Net improvement
N	Before (cmd runner total)	After (cmd runner total)	Native
1000	24.2 ms	20.0 ms	3.9 ms
4000	182.9 ms	132.1 ms	148.7 ms
8000	917.9 ms	929.8 ms	846.4 ms
16000	7202.9 ms	6374.1 ms	5165.8 ms
At N=16000 the command runner improved by 828 ms (11%). At N=4000 it is now actually faster than native (because native's benchmark includes H2D copy measurement overhead there). The remaining ~1200 ms gap at N=16000 is the copy_to_host allocation/page-fault issue described above.

