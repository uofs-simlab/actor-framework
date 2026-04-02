# CAF-CUDA Test Analysis

## Overview

This document analyses the fairness of the existing benchmark tests in `runtime-overhead-comparison` and `sequence-of-independent-tasks`, describes tweaks needed to ensure fair measurement, proposes additional tests to strengthen the SC paper, and recommends figure layouts for each.

---

## Test 1: `runtime-overhead-comparison`

### What is being tested
Single-shot matrix-multiplication executions of increasing problem size (N = 1000, 2000, 4000, 8000, 16000). The intent is to compare end-to-end wall-clock cost (alloc → H2D → kernel → D2H) across three execution models: native CUDA driver API, CAF actor-facade, and CAF command-runner.

### Fairness issues found

| # | Issue | Affected variants | Impact |
|---|-------|-------------------|--------|
| 1 | **No warmup run** | actor-facade, command-runner | CUDA lazy-init (context creation, JIT compilation, cubin load) is rolled into the first timed data point, inflating it by hundreds of milliseconds. Native has an explicit warmup at N=64, actors do not. |
| 2 | **Integer ms timer resolution** | actor-facade, command-runner | Both use `std::chrono::duration_cast<std::chrono::milliseconds>` which truncates to integer ms. Native uses `std::chrono::duration<double, std::milli>`. For small N the true duration may be < 1 ms, yielding a reported time of 0 ms. |
| 3 | **Per-invocation cubin loading** | actor-facade, command-runner | The actor-facade calls `spawnFromCUBIN` (spawns a new GPU actor backed by a fresh cubin load) and the command-runner calls `create_program_from_cubin` inside the timed message handler on every invocation. Native loads the cubin once before the timing loop. This means actor tests include cubin-load and GPU-actor-spawn overhead in every measurement, while native does not. For an overhead paper this must be explicitly disclosed. |
| 4 | **Memory deallocation outside native timer** | native (difference vs actors) | In `main_cuda_native.cpp`, `t_total_end` is captured immediately after `cuStreamSynchronize` for the D2H copy; `cuMemFree` and `cuStreamDestroy` occur after `t_total_end`. In actor tests the equivalent cleanup is implicit and not separately timed. Native total therefore excludes free cost while actor totals may include it. |
| 5 | **Async enqueue vs wall-clock breakdown** | native | H2D A, H2D B, and kernel-launch timers in the native code capture only the time to *enqueue* the async operation, not GPU execution time. These per-phase numbers are misleading (all near zero). Only the TOTAL (which spans through `cuStreamSynchronize`) accurately reflects wall-clock cost. Document this in any paper table. |

### Recommended tweaks

1. **Add warmup to actor tests** — before the `sizes` loop, perform one `N=64` invocation and discard its result:
   ```cpp
   // In caf_main, before the sizes loop:
   {
     int warmup_N = 64;
     auto w = self->spawn(caf::actor_from_state<MatMult>, warmup_N);
     self->mail(warmup_N).request(w, caf::infinite).receive(
       [](double) {}, [](const caf::error&) {});
   }
   ```

2. **Fix timer resolution in actor tests** — replace:
   ```cpp
   double duration = std::chrono::duration_cast<
       std::chrono::milliseconds>(end_ - start_).count();
   ```
   with:
   ```cpp
   double duration = std::chrono::duration<double, std::milli>(end_ - start_).count();
   ```

3. **Make cubin loading consistent** — either:
   - Move `create_program_from_cubin` / `spawnFromCUBIN` outside the timed window (into the actor constructor) for a "steady-state" comparison; OR
   - Add a note in the paper that cubin loading is included in actor timings as a one-time dispatch cost and is amortised over repeated runs.
   Recommended: cache `program` in actor state (constructor), pass `dim` separately to the handler, so the timed window only covers H2D → kernel → D2H.

4. **Align timer scope** — ensure all three variants time exactly: (device alloc) + H2D + kernel wait + D2H. The native test excludes `cuMemFree`; actor tests should also note whether cleanup is measured.

---

## Test 2: `sequence-of-independent-tasks`

### What is being tested
Repeated independent matrix-multiply submissions at fixed N=1000 to measure cumulative overhead over increasing numbers of iterations (up to 10 000). This exercises actor-framework scheduling, message-passing, and GPU dispatch latency at scale.

### Fairness issues found

| # | Issue | Affected variants | Impact |
|---|-------|-------------------|--------|
| 1 | **No warmup** | all three | CUDA lazy-init hits the first iteration and skews the first data point. |
| 2 | **Output structure mismatch (now fixed)** | actor-facade, command-runner | Before this fix: actors printed per-iteration duration and a final total; native printed cumulative time at series checkpoints {1000, 2000, …, 10000}. This made direct plotting impossible. **Fixed:** both actor examples now print `[SERIES RESULT]` lines every 1000 iterations using wall-clock time from loop start, matching the native format exactly. |
| 3 | **Verbose per-iteration stdout** | actor-facade, command-runner (before fix) | `self->println("Duration = {}", ...)` on every iteration forces a synchronous log write in the CAF scheduler, adding measurable overhead to each iteration and dwarfing the actual GPU time. **Fixed** by removing per-iteration prints. |
| 4 | **New actor spawned per iteration** | actor-facade, command-runner | Both actor tests create a fresh `MatMult` actor (including N×N vector allocation, RNG initialisation) for every iteration. The native test uses **persistent** host buffers (`h_a`, `h_b`, `h_c` allocated once). Actor construction is *outside* the per-iteration timer but the spawn syscall and memory allocation add real wall-clock cost between iterations. |
| 5 | **Data re-initialisation** | actor-facade, command-runner | Each new `MatMult` actor re-runs the RNG fill (same seed, same values), adding CPU work between GPU tasks. Native fills once. |
| 6 | **Stream synchronisation placement** | native | Native synchronises the stream only **once per series** (after all iterations), meaning GPU work pipelines across iterations and overlaps with CPU overhead. Actor tests synchronise after every iteration (request/receive is blocking). This makes the native test faster in an unrealistic way for this comparison and should be documented. |
| 7 | **Wall-clock vs. summed-GPU-time** | actor-facade, command-runner (after fix) | The fixed actor tests use wall-clock time from loop start, which includes actor-framework overhead (spawn, message passing, scheduling). This is intentional and correct for an overhead comparison — it measures the true end-to-end cost visible to the user. |

### Recommended tweaks

1. **Add warmup** to all three variants before the timed loop.

2. **Reuse actor state** — instead of spawning a new actor each iteration, spawn one `MatMult` actor and send it multiple sequential messages. This isolates GPU dispatch overhead from actor-construction overhead:
   ```cpp
   auto worker = self->spawn(caf::actor_from_state<MatMult>, N);
   auto loop_start = clock_t_::now();
   for (int i = 0; i < iterations; i++) {
     self->mail(N).request(worker, caf::infinite).receive(...);
     if ((i + 1) % 1000 == 0) { /* print checkpoint */ }
   }
   ```

3. **Match native's per-series structure optionally** — either run all 10 000 iterations continuously (current approach after fix), or mirror native's "restart from zero for each series length" approach for exact data parity. The continuous approach is simpler and comparably valid.

4. **Document the native stream-pipeline advantage** in the paper — native's single-sync-per-series means GPU can be working on iteration i+1 while CPU handles iteration i. Actor tests necessarily synchronise per iteration due to blocking request/receive.

---

## Proposed Additional Tests

The following tests are recommended to make a stronger SC paper. Each measures a different dimension of the CAF-CUDA value proposition.

---

### Test 3: Kernel Dispatch Latency (Microbenchmark)

**Motivation:** Isolate the pure framework overhead introduced by the actor system. Use a trivial "no-op" or minimal-work kernel to remove GPU compute time from the equation. The measured time is purely the framework's dispatch path.

**Methodology:**
- Kernel: 1-thread kernel that writes a single constant to a 1-element output buffer.
- Variants: native CUDA driver API, actor-facade, command-runner.
- Measurement: 10 000 iterations. Record cumulative time every 1000 iterations. Compute mean dispatch latency per call (elapsed / iterations).
- Report: mean ± std-dev of per-call latency in microseconds.

**Why this matters:** It gives a clear, kernel-independent number for "what does CAF-CUDA cost per GPU call?" which directly supports an overhead claim in the paper.

**Figure:** Bar chart (one bar per variant, mean latency in µs, with error bars for std-dev).

---

### Test 4: Scalability with Concurrent Independent Tasks (Throughput)

**Motivation:** The native CUDA driver API is single-threaded by default. The actor model naturally expresses concurrent GPU dispatch across multiple logical workers. This test measures whether CAF-CUDA can achieve higher GPU throughput on wide machines by running multiple concurrent actor-GPU pipelines.

**Methodology:**
- Spawn P = {1, 2, 4, 8} actor workers, each submitting independent matrix-multiply tasks in a loop.
- Each worker handles 2500 tasks (total = P × 2500 = 10 000 tasks regardless of P).
- Compare wall-clock completion time for the full 10 000 tasks as P increases.
- Native baseline: single-threaded sequential loop.
- CAF variant: P event-based actors dispatching concurrently, CAF scheduler managing them.

**Why this matters:** This showcases a core actor-model advantage — expressing concurrent GPU work with zero explicit synchronisation. If throughput increases with P, CAF-CUDA provides a practical productivity benefit over hand-written concurrent CUDA code.

**Figure:** Line chart — x-axis: number of concurrent actor workers (P), y-axis: total wall-clock time for 10 000 tasks (ms). Two lines: CAF-CUDA vs. native threaded.

---

### Test 5: Pipeline of Dependent Kernel Stages

**Motivation:** Real HPC workloads chain GPU kernels (e.g., preprocess → compute → postprocess). The actor model expresses this as a message-passing pipeline with no explicit sync primitives. Measure the overhead relative to a manually-coded CUDA pipeline.

**Methodology:**
- 3-stage pipeline: Stage 1 (element-wise scale), Stage 2 (matrix multiply), Stage 3 (element-wise reduce/sum).
- Native: three sequential `cuLaunchKernel` calls with one stream, one final sync.
- CAF-CUDA actor chain: three actors, each forwarding results to the next via `mail(...).request(...).then(...)`.
- Metric: end-to-end latency for N = {512, 1024, 2048, 4096}; run 100 times, report mean.

**Why this matters:** Shows how naturally the actor model expresses dataflow DAGs and quantifies the overhead of the message-passing stitching between stages.

**Figure:** Grouped bar chart — x-axis: matrix size N, y-axis: end-to-end pipeline latency (ms). Two bars per N: native vs. CAF-CUDA actor chain. Include a "framework overhead %" annotation above each pair.

---

### Test 6: Error Recovery / Fault Tolerance

**Motivation:** Production GPU applications must handle occasional kernel failures (out-of-memory, illegal access, etc.). This test demonstrates the practical advantage of CAF's supervision model over manual `cudaError_t` checking.

**Methodology:**
- Inject a fault (e.g., provide a null pointer or invalid grid size) in 1% of kernel dispatches.
- Measure: (a) lines of boilerplate error-handling code (static metric), (b) time to detect and recover from the error (dynamic metric) in both native and CAF-CUDA variants.
- Native: requires manual check after every CUDA call + manual retry or abort logic.
- CAF-CUDA: default-handler or supervisor catches error atom and retries.

**Why this matters:** Quantifies the engineering productivity benefit. For a systems paper, showing that error handling is automatic and adds minimal overhead is a strong usability claim.

**Figure:** Code snippet side-by-side (native vs. CAF-CUDA, highlighted brevity) plus a small bar chart comparing time-to-recovery (ms) and lines-of-error-handling-code.

---

### Test 7: Memory Pressure and Transfer Bandwidth

**Motivation:** For large matrices, PCIe transfer time dominates. This test separates compute overhead from transfer overhead and shows how the CAF-CUDA transfer API compares to native async memcpy bandwidth.

**Methodology:**
- Fix N and vary data sizes: 256 MB, 512 MB, 1 GB transfers.
- Variants: native async `cuMemcpyHtoDAsync`, CAF-CUDA in-arg/out-arg.
- Measure: achieved H2D and D2H bandwidth (GB/s) for each variant.
- Use `cudaEvent` or high-resolution CPU timers around memcpy+sync.

**Why this matters:** If CAF-CUDA achieves near-native memcpy bandwidth, it validates that the abstraction layer does not add data-path overhead.

**Figure:** Dual bar chart — one for H2D bandwidth, one for D2H bandwidth. X-axis: transfer size; y-axis: GB/s. Two bars per size: native vs. CAF-CUDA.

---

### Test 8: Actor Model Composability — Multi-GPU Orchestration

**Motivation:** One of the strongest arguments for CAF-CUDA is that multi-GPU work decomposition becomes trivially composable using actors. This test shows that scaling to multiple GPUs requires no architectural change to CAF-CUDA code.

**Methodology:**
- System with 2+ GPUs. Task: multiply four independent matrix pairs.
- Native: manual `cudaSetDevice` calls to route work to each GPU, manual thread management.
- CAF-CUDA: one actor per GPU, coordinator actor fan-out the four tasks and `await` all results.
- Measure: wall-clock completion time as GPU count increases from 1 to available maximum.

**Why this matters:** Demonstrates linear or near-linear speedup with GPU count using the actor model with zero synchronisation primitives. This is a key selling point for HPC.

**Figure:** Line chart — x-axis: number of GPUs; y-axis: wall-clock time for 4×matmul (ms). Two lines: native (manual) vs. CAF-CUDA actor fan-out. Include an "ideal linear speedup" reference line.

---

## Figure Presentation Summary

| Test | Recommended Figure Type | Key Axes | What to Highlight |
|------|-----------------------|----------|-------------------|
| T1: runtime-overhead-comparison | Line chart | x: matrix size N; y: time (ms) | Convergence of actor overhead to native at large N |
| T2: sequence-of-independent-tasks | Line chart | x: iterations (1000–10000); y: cumulative time (ms) | Slope difference = per-task overhead |
| T3: dispatch latency | Bar chart | x: variant; y: mean latency (µs) ±σ | Raw framework dispatch cost |
| T4: concurrent scalability | Line chart | x: concurrent workers; y: total time (ms) | Throughput advantage of actor concurrency |
| T5: pipeline stages | Grouped bar | x: matrix size N; y: pipeline latency (ms) | Overhead % annotation per pair |
| T6: error recovery | Code snippet + bar | N/A + time-to-recover | Actor brevity vs. native verbosity |
| T7: memory bandwidth | Dual bar | x: transfer size; y: GB/s | Bandwidth parity validates no data-path overhead |
| T8: multi-GPU scaling | Line chart | x: GPU count; y: time (ms) | vs. ideal linear speedup |

### General presentation advice

- **Use log-scale y-axis** for tests T1 and T3 if the range spans more than one order of magnitude.
- **Error bars** (mean ± std-dev over ≥ 5 runs) on all dynamic timing figures are mandatory for a credible systems paper.
- **Overhead percentage annotation**: for each comparison figure, add a secondary y-axis or annotation showing `((actor − native) / native) × 100%`. This directly answers "how much does CAF-CUDA cost?"
- **Normalise to native baseline** in a summary figure: one chart with bars for each test showing relative overhead (%). A value close to zero means the actor abstraction is transparent.
- **Table 1: static code complexity** — count lines-of-code and explicit synchronisation calls for each variant (native / actor-facade / command-runner) across tests T1–T5. Fewer lines + zero explicit sync in CAF-CUDA is a strong usability argument.

---

## Results Analysis: Why the Performance Gap Is So Large

*(Added after reviewing `resutls.txt` / `results.txt` and tracing the full CUDA dispatch path.)*

### Observed numbers

**runtime-overhead-comparison (single-shot, N=1000):**
| Variant | Time (ms) |
|---------|-----------|
| Native CUDA | 3.74 |
| Command-runner | 19.34 |
| Actor-facade | 29.69 |

**sequence-of-independent-tasks (1000 iterations, N=1000):**
| Variant | Cumulative time (ms) | Per-iteration (ms) |
|---------|---------------------|-------------------|
| Native CUDA | 3891 | 3.89 |
| Command-runner | 18,643 | 18.64 |
| Actor-facade | 18,168 | 18.17 |

---

### Root cause 1: The native sequence test pipelines GPU work — actors cannot (primary cause)

**This is the single biggest source of the gap** and it is a structural difference, not a bug.

The native `sequence-of-independent-tasks` test calls `cuMemFree` with the CUDA Driver API, which in CUDA 12 is **non-blocking** (logically deferred). The memory is reclaimed by CUDA once the GPU finishes using it, but the CPU returns immediately and loops to the next iteration. The GPU stream therefore has all 1000 iterations queued back-to-back with no gap:

```
GPU stream: [H2D A1][H2D B1][kernel1][D2H C1][H2D A2][H2D B2][kernel2][D2H C2]...
CPU:        loop() ← returns immediately for each iteration
cuStreamSynchronize: called ONCE at iteration 1000
```

The GPU is at **100% utilization** throughout. The measured time is pure GPU throughput: 1000 × 3.89 ms.

The actor tests call `cuStreamSynchronize` **inside `copy_to_host()`** every single iteration (via `collect_output_buffers → mem_ref::copy_to_host`). The GPU is idle while the CPU does CAF message routing, actor scheduling, and memory management before the next iteration begins:

```
[H2D A][H2D B][kernel][D2H C][sync] ← GPU idle → [H2D A][H2D B]...
                                      14 ms gap
```

GPU utilization in actor tests ≈ 3.89 / 18.17 = **21%**.

**This is the dominant overhead source**: 14.3 ms of GPU idle time per iteration out of 18.17 ms total.

---

### Root cause 2: actor-facade serializes input data through CAF messages (extra copies)

When `actor_facade` sends data to the GPU actor using `self_->mail(create_in_arg(A_), ...).request(gpuActor_, ...)`, CAF serializes the arguments into a message. The serialization path calls `in_impl<T>::get_buffer()`:

```cpp
// in types.hpp:
std::vector<T> get_buffer() const {
    return std::vector<T>(ptr_, ptr_ + size_);  // full NxN copy
}
// in global.hpp inspect():
auto buf = x.get_buffer();  // copies 4 MB
f.object(x).fields(f.field("buffer", buf));  // writes to message
```

The receiving GPU actor then deserializes, reading a new `std::vector<T>` back from the message. This is a **full round-trip copy of A and B through CAF's message store** — 8 MB of CPU memory per iteration for N=1000.

The command-runner does not have this problem because the data path is a direct function call with no message passing. This is why command-runner (18.6 ms) is consistently faster than actor-facade (18.2 ms) for small N, and explains the additional overhead in actor-facade at the single-shot level (29.7 ms vs 19.3 ms).

---

### Root cause 3: `collect_output_buffers → copy_to_host` allocates a new heap vector every call

```cpp
// In mem_ref.hpp copy_to_host():
std::vector<T> host_data(num_elements_);  // allocates 4 MB every call
cuMemcpyDtoHAsync(host_data.data(), memory_, bytes, s);
cuStreamSynchronize(s);
return host_data;  // returns by value → moves to output_buffer
```

Native writes D2H to a **persistent** pre-allocated `h_c` buffer. The actor path allocates 4 MB → does D2H → returns (moves) the vector → gets placed in an `output_buffer` → then `extract_vector<int>` copies it again to return to the caller. This is two 4 MB allocations/copies that native does not do.

For N=1000 at ~30 GB/s memory bandwidth: ~0.3 ms per dispatch. Minor compared to the GPU idle time but accumulates over 10 000 iterations (≈ 3 seconds overhead just from these copies).

---

### Root cause 4: Multiple `cuCtxPushCurrent` / `cuCtxPopCurrent` per dispatch

The actor dispatch path calls the context push/pop pair in:
1. `global_argument` for A (H2D A: push+pop)
2. `global_argument` for B (H2D B: push+pop)
3. `scratch_argument` for C (alloc: push+pop)
4. `launch_kernel_internal` (kernel launch: push+pop)
5. `copy_to_host` in `collect_output_buffers` (D2H + sync: push+pop)

That is **5 context switch pairs** per dispatch vs. 0 in native (context is already current on the calling thread). Each `cuCtxPushCurrent` involves a kernel ioctl, contributing to the elevated `sys` time seen in `time` output:

- Native `sequence-of-independent-tasks` (10 000 iter): sys = 26 s → 2.6 ms/iter syscall overhead
- Actor-facade (10 000 iter): sys = 84 s → 8.4 ms/iter syscall overhead
- Command-runner (10 000 iter): sys = 111 s → 11.1 ms/iter syscall overhead (more context ops than facade for this test)

The extra syscall time alone accounts for ~6–8 ms/iteration beyond native.

---

### Why the old `mmul-actor-benchmarking` tests showed closer performance

The old `baseline-comparison/actors` benchmark used `anon_mail(...).send(a)` (fire-and-forget) instead of blocking `request/receive`:

```cpp
// Old benchmark — all iterations queued non-blocking:
for (int i = 0; i < iterations; i++)
    anon_mail(matrixA, matrixB, matrix_size).send(a);
sys.await_all_actors_done();  // single barrier at end
```

This is structurally identical to the native test's pipelining: all iterations are queued to the actor's mailbox with no blocking. The actor processes them sequentially from its mailbox just as the GPU stream processes native ops sequentially. GPU utilization was near 100% in both cases.

The current `sequence-of-independent-tasks` uses blocking `request/receive` per iteration, which was the correct change for getting per-iteration results but introduced per-iteration GPU stalls. The old benchmarks were not broken because of API changes to improve results — they were doing something fundamentally different (throughput-mode operation).

---

### Is the native test "cheating"?

**No** — but the two tests are measuring different things:

- Native `sequence` test measures **GPU throughput**: how long does the GPU take to run 1000 matrix multiplies back-to-back with no idle time. Answer: ~3.89 ms × 1000 = 3890 ms.
- Actor tests measure **end-to-end application throughput** including the framework scheduling: how long does it take for a program to dispatch 1000 work units and collect all results using the actor model. Answer: ~18 ms × 1000 = 18 000 ms.

Neither is wrong. But they need to be shown as measuring different things in the paper.

---

### How to get a truly fair apples-to-apples sequence comparison

**Option A — Serialised latency (fairest overhead measure):**
Add `cuStreamSynchronize(stream)` after each `cuMemcpyDtoHAsync` in the native sequence test's inner loop, forcing native to also complete one full GPU round-trip per iteration before queuing the next. This matches actor semantics exactly and isolates the pure framework overhead. Expected result: native ≈ 3.9 ms/iter, actors ≈ 6–8 ms/iter (once `extract_vector` cost is also removed). Overhead ratio: ~1.5–2x rather than ~4.5x.

**Option B — Throughput mode with fire-and-forget actors:**
Restructure actor tests to use `anon_mail(...).send(worker)` for all iterations, then `sys.await_all_actors_done()`. This matches the native pipelining and shows the actor model can achieve near-native throughput when results are not needed per-iteration. This recovers the closer-to-native performance seen in the old benchmarks.

Both options should be presented in the paper: Option A as the honest latency overhead figure, Option B as the throughput figure demonstrating the actor model's practical ceiling.

---

### Additional code-level improvements that would reduce overhead

| Item | Current behaviour | Improvement | Expected gain |
|------|------------------|-------------|---------------|
| `in<int>` serialization in actor-facade | Full N×N copy through CAF message store | Mark `in<int>` as `CAF_ALLOW_UNSAFE_MESSAGE_TYPE` to pass by shared pointer for intra-process messages | Eliminates ~8 MB copy per dispatch for actor-facade |
| `copy_to_host` allocation | Allocates new `std::vector` every call | Add `copy_to_host(T* dst, size_t n)` overload writing to caller-provided buffer | Eliminates one 4 MB allocation per dispatch |
| `extract_vector` in test code | Copies 4 MB unnecessarily (result discarded) | Already removed from current test code — confirm not re-introduced | Eliminates one 4 MB copy per dispatch |
| Context push/pop | 5 round-trips per dispatch | Cache context as thread-local current once at actor startup | Eliminates ~5–8 ms/iter sys overhead |
| Per-iteration `cuMemAlloc` | 3 driver-API allocations per dispatch | Use stream-ordered allocator (`cuMemAllocAsync`/`cuMemFreeAsync`) or a device memory pool | Could cut allocation overhead by 10–100x |

The most impactful single change would be adding per-iteration `cuStreamSynchronize` to the native sequence test (Option A above) and then presenting the remaining gap as honest framework overhead.

---

### Summary table: overhead sources

| Source | Overhead (ms/iter, N=1000) | Affects |
|--------|---------------------------|---------|
| GPU idle time (CAF scheduling between iterations) | ~6–10 | Both actor variants |
| Extra syscalls (`cuCtxPushCurrent/Pop` ×5) | ~6–8 | Both actor variants |
| Serialization copy of A+B through CAF message | ~0.3 | Actor-facade only |
| `copy_to_host` vector allocation + D2H copy in place | ~0.3 | Both actor variants |
| CAF message routing (send + receive) | ~0.2 | Both actor variants |
| **Total (estimated)** | **~13–19 ms** | **Both; facade slightly worse** |

The numbers match the observed ~14.3 ms/iter gap. At N=4000, the GPU kernel itself takes ~146 ms so the ~14 ms framework overhead becomes < 10%, which is why command-runner approaches native performance at large N in the runtime-overhead test.
