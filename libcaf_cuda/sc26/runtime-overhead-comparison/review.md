# Runtime Overhead Comparison — Review

## 1. Test Fairness Audit

### Is the output vector being hidden behind `std::move`?

**Short answer: No — the data copy is real and is not hidden.**

The `std::move` in the command-runner benchmark only transfers *ownership* of an already-host-resident `std::vector<int>`.  The actual Device-to-Host (D2H) transfer occurs inside `runner.run()` via `mem_ref::copy_to_host()`, which issues `cuMemcpyDtoHAsync` followed by `cuStreamSynchronize`.  By the time `extract_vector(std::move(result_buffer))` executes, the data has already been copied to host RAM; the move simply takes the existing heap allocation without a second copy.  Both the command-runner and the native benchmark end the timed section with a fully host-accessible `std::vector<int>`.

### Structural asymmetry that *does* affect fairness

Even though no data is being hidden, there is one genuine asymmetry between the two programs:

| Operation | Native (`main_cuda_native`) | Command-runner (`main_actor_command_runner`) |
|---|---|---|
| Host output buffer (h_c / host_data) | Allocated **before** `t_total_start` — pages are backed | Allocated **inside** `runner.run()` in `copy_to_host()` — pages faulted during the timed window |
| Device memory free | Measured **after** `t_total_end`, excluded from total | Happens **inside** `runner.run()` when `mem_refs` go out of scope at end of `command::enqueue()` |

**Host output buffer allocation (`copy_to_host` invariant):**
`mem_ref::copy_to_host()` always does `std::vector<T> host_data(num_elements_)`.  For `int`, this value-initialises every element to zero, which on Linux demand-paging causes the OS to back every page.  For N=1000 that is ~4 MB (≈1 024 pages); for N=16 000 it is ~1 GB (≈262 144 pages).  The native program pre-initialises `h_c(elements)` before starting the timer, so those page faults are invisible to native's clock.

**Device memory free:**
`mem_ref::reset()` calls `cuMemFree` when the intrusive-pointer reference count drops to zero.  The three `mem_refs` for A, B, and C are destroyed inside `command::enqueue()` once `collect_output_buffers` has consumed them.  This entire path sits within `runner.run()` and is therefore inside the timed window.  The native benchmark measures `cuMemFree` separately *after* `t_total_end` and excludes it from the total.

**Recommendation to restore symmetry:**
1. The command-runner test should accept a user-supplied output buffer (using the `copy_to_host(T* dst, count)` overload that already exists on `mem_ref`) so the allocation occurs outside the timed window — mirroring native's pre-allocated `h_c`.
2. Either include `cuMemFree` in the native total, or restructure the actor code to defer device memory freeing until after `t_total_end`.  The simpler fix is to add `cuMemFree` to native's `t_total_end` boundary.

---

## 2. Runner.run() Bottleneck Analysis

The benchmark results (from the attached output files) show:

| N | Native (ms) | Command-runner (ms) | Delta |
|---|---|---|---|
| 1000 | 3.84 | 19.39 | **+15.5 ms** |
| 2000 | 18.51 | 47.35 | **+28.8 ms** |
| 4000 | 148.28 | 138.45 | −9.8 ms (actor wins) |
| 8000 | 847.58 | 661.87 | −185.7 ms (actor wins) |
| 16000 | 5141.01 | 6752.28 | **+1611 ms** |

At small N the actor overhead dwarfs the actual work.  At large N the overhead is either masked or interacts with a different memory-transfer regime; the picture inverts at N=16 000 when the output-buffer page-fault cost (~1 200 ms, per `overhead.md`) begins to dominate again.

### Bottleneck 1 — Output buffer allocation + page faults inside the timed window

**Location:** `mem_ref::copy_to_host()` → `std::vector<T> host_data(num_elements_)`

Every call to `runner.run()` allocates a fresh, zero-initialised host vector to receive the D2H data.  On Linux, new anonymous memory pages are demand-faulted: the kernel must allocate physical pages and zero-fill them on first access.  For pageable DMA, the CUDA driver cannot begin the actual transfer until those pages are accessible, so the faults add *serialised* CPU time directly before the DMA.

Estimated cost (from `overhead.md`): **~4.7 ms at N=1000**, **~1 200 ms at N=16 000**.

Native avoids this entirely because `h_c` is pre-allocated and its pages are backed before the timer starts.

**Proposed fix:**  
Extend the `command_runner::run()` API to accept a pre-allocated output span or pointer:

```cpp
// New overload — caller provides an already-faulted output buffer
template <class... Us>
auto run(program_ptr program,
         nd_range dims,
         caf::actor_id actor_id,
         T* output,          // pre-allocated destination
         size_t output_count,
         Us&&... xs);
```

Internally this calls the existing `copy_to_host(T* dst, size_t count)` overload on `mem_ref`, bypassing the allocation.  No semantic change — the caller retains full ownership.  For the benchmark, simply declare the output vector before `t_total_start` and pass its `.data()` pointer into `runner.run()`.

---

### Bottleneck 2 — Repeated `cuCtxPushCurrent` / `cuCtxPopCurrent` for every GPU operation

**Location:** `device::global_argument()`, `device::scratch_argument()`, `device::launch_kernel_mem_ref()`, `mem_ref::copy_to_host()`

For a single kernel launch the call sequence looks like:

```
global_argument(A):    cuCtxPushCurrent ── cuMemAlloc ── cuMemcpyHtoDAsync ── cuCtxPopCurrent
global_argument(B):    cuCtxPushCurrent ── cuMemAlloc ── cuMemcpyHtoDAsync ── cuCtxPopCurrent
scratch_argument(C):   cuCtxPushCurrent ── cuMemAlloc ── cuCtxPopCurrent
launch_kernel_mem_ref: cuCtxPushCurrent ── cuLaunchKernel ── cuCtxPopCurrent
copy_to_host(C):       cuCtxPushCurrent ── cuMemcpyDtoHAsync ── cuStreamSynchronize ── cuCtxPopCurrent
```

That is **5 push/pop pairs** — 10 driver API calls — per kernel invocation, for work that could be done inside a single push/pop bracket.  Each `cuCtxPushCurrent` acquires a driver-internal TLS lock and may flush pending work; `cuCtxPopCurrent` releases it.  On modern NVIDIA drivers each round-trip costs roughly 5–50 µs, but under any system load it can spike.  With 10 calls that is an estimated **0.05–0.5 ms** fixed overhead per run, compounding with the other issues.

The native benchmark never calls `cuCtxPushCurrent` / `cuCtxPopCurrent` inside the timed window; the context created by `cuCtxCreate` remains implicitly current on the calling thread until the program exits.

**Proposed fix:**  
Bracket the *entire* `enqueue()` execution path with a single push/pop, and remove the individual push/pops from `global_argument`, `scratch_argument`, and `copy_to_host`.  Concretely, `launch_kernel_mem_ref` already knows which device context to use; push it once at the top, perform all allocations, H2D copies, kernel launch, D2H copy, and free in sequence, then pop.

```cpp
// In command::enqueue() (or base_command::base_enqueue()):
CHECK_CUDA(cuCtxPushCurrent(dev_->getContext()));
// ...all allocations, copies, kernel, collect_output_buffers...
CHECK_CUDA(cuCtxPopCurrent(nullptr));
```

This is safe because all operations target the same device and stream.  The inner helpers must then be refactored into variants that assume the context is already current (i.e., drop the push/pop from `global_argument_unchecked`, etc.).

---

### Bottleneck 3 — Device memory freed inside the timed window

**Location:** `mem_ref::~mem_ref()` → `reset()` → `cuMemFree`, triggered when the `mem_refs` tuple in `command::enqueue()` goes out of scope after `collect_output_buffers`.

At N=1000 native measures device free at **~0.24 ms**; at larger N this grows.  The actor benchmark includes this cost inside `runner.run()` while the native benchmark excludes it.

Beyond the benchmark asymmetry, allocating and freeing device memory every kernel launch prevents the CUDA driver from reusing allocations.  `cuMemAlloc`/`cuMemFree` involve driver-side bookkeeping (virtual address space reservation, free-list management) that adds non-trivial latency.

**Proposed fix (two independent improvements):**

1. **Benchmark fix:** Record `t_total_end` immediately after `cuStreamSynchronize` in `collect_output_buffers` / `copy_to_host`, before `mem_refs` go out of scope and trigger `cuMemFree`.  This mirrors what native does.

2. **Structural fix (device memory pooling):** Introduce a per-device arena allocator or reuse a `CUmemoryPool` (available since CUDA 11.2 via `cuMemAllocAsync` / `cuMemFreeAsync`).  When the same actor calls `runner.run()` for the same problem size repeatedly, the allocations for A, B, C would be served from already-committed device memory with no driver roundtrip.  `cuMemAllocAsync` schedules the allocation on the stream and returns instantly; `cuMemFreeAsync` returns the memory to the pool without synchronising the CPU.

---

### Bottleneck 4 — `prepare_kernel_args` allocates one `new CUdeviceptr` per GPU argument

**Location:** `device::prepare_kernel_args()` → `new CUdeviceptr(mem->mem())`

For a 4-argument kernel (A, B, C, N) this performs **3 heap allocations** (N is scalar and uses a stack pointer) plus 3 `delete`s in `cleanup_kernel_args`.  These are tiny individually, but they happen under the context lock and add pointer-chasing to the hot path.

**Proposed fix:**  
Use a small stack-allocated array (or `std::array` / VLA) sized to the argument count, which is known at compile time via the template parameter pack `Ts...`:

```cpp
// Replace the heap-allocated vector with a compile-time-sized array:
template <typename... Ts>
kernel_arg_pack prepare_kernel_args(const std::tuple<Ts...>& args) {
    constexpr size_t N = sizeof...(Ts);
    std::array<CUdeviceptr, N> dev_ptrs;  // stack, no heap allocation
    std::array<void*, N> ptrs;
    // fill ptrs from mem_refs...
    return {ptrs, dev_ptrs};  // or inline the launch to avoid returning at all
}
```

Since `Ts...` is a template parameter pack, `sizeof...(Ts)` is a compile-time constant and the array lives on the stack.

---

### Bottleneck 5 — Per-invocation `command<>` heap allocation

**Location:** `command_runner::run()` → `caf::make_counted<command_t>(...)`

Every call to `runner.run()` heap-allocates a refcounted `command<Actor, Ts...>` object.  Most of the object's data (program, dims, device) is the same across calls if the problem parameters don't change.  Under high kernel-launch rates this adds GC pressure.

**Proposed fix:**  
Pre-construct the `command<>` once in the `command_runner` constructor (or a `prepare()` step) and cache it, resetting only the per-invocation arguments (the kernel args tuple) before each `enqueue()`.  Alternatively, remove the heap allocation entirely by inlining the logic of `base_enqueue()` directly into `command_runner::run()` without a separately-allocated command object.

---

## 3. Summary Table

| # | Bottleneck | Affected N | Estimated Impact | Proposed Fix |
|---|---|---|---|---|
| 1 | Output buffer alloc + page faults in `copy_to_host` | All, severe at N≥1000 | ~4.7 ms @ N=1000, ~1200 ms @ N=16000 | Accept pre-allocated output buffer in `runner.run()` |
| 2 | 5 `cuCtxPushCurrent`/`cuCtxPopCurrent` pairs per launch | All, dominant at small N | ~0.05–0.5 ms (variable) | Single push/pop wrapping entire `enqueue()` |
| 3 | `cuMemFree` inside timed window | All | ~0.24 ms @ N=1000 | Async pool (`cuMemAllocAsync`/`cuMemFreeAsync`) + timer boundary fix |
| 4 | `new CUdeviceptr` per kernel argument | All | < 0.05 ms | Stack-allocated argument array |
| 5 | Per-run `command<>` heap allocation | All | < 0.05 ms | Cache or inline the command object |

Fixing bottleneck 1 alone would likely close the majority of the gap at N=1000.  Fixing bottleneck 2 (consolidating context push/pop) would further reduce the fixed overhead.  Fixing bottleneck 3 properly (async pool allocations) would make the actor framework competitive with or faster than the raw driver API at all matrix sizes, since it would eliminate both the page-fault and the `cuMemFree` costs.  Bottlenecks 4 and 5 are minor polish items.
