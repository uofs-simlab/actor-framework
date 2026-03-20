# libcaf_cuda Code Review & Roadmap

## Table of Contents

- [Executive Summary](#executive-summary)
- [The Good](#the-good)
- [The Bad](#the-bad)
- [The Ugly](#the-ugly)
- [What to Keep](#what-to-keep)
- [What to Change](#what-to-change)
- [Roadmap](#roadmap)
  - [Phase 1 — Safety & Correctness](#phase-1--safety--correctness)
  - [Phase 2 — CAF Integration](#phase-2--caf-integration)
  - [Phase 3 — API Ergonomics](#phase-3--api-ergonomics)
  - [Phase 4 — Runtime API Migration](#phase-4--runtime-api-migration)
- [Actionable Task Descriptions](#actionable-task-descriptions)

---

## Executive Summary

The libcaf_cuda module is a substantial piece of work that brings GPU computing
into the CAF actor model. The core concept — wrapping CUDA kernel launches behind
actor-friendly abstractions with tagged memory types (`in<T>`, `out<T>`,
`in_out<T>`) — is sound and genuinely useful. There is working multi-GPU
support, stream pooling, an advanced scheduling/control layer, and a test suite
that exercises real kernels.

However, the implementation has significant issues at multiple levels: safety
bugs that can cause silent data corruption, architectural decisions that violate
CAF's design principles, code quality problems that make maintenance difficult,
and API design choices that create footguns for users. Below is a thorough
analysis organized into the good, the bad, and the ugly, followed by a
prioritized roadmap.

---

## The Good

These are the aspects that demonstrate solid design thinking and should be
preserved and built upon.

### 1. Tagged Memory Type System (`in<T>`, `out<T>`, `in_out<T>`)

The wrapper types that annotate kernel arguments with their access direction are
the single best design decision in the codebase. They:

- Make kernel argument intent explicit at the call site
- Enable the framework to automatically manage GPU memory allocation
- Allow the device class to differentiate input-only, output-only, and
  bidirectional buffers at compile time
- Support both scalar and buffer modes transparently

The `raw_t<T>` type trait for unwrapping is clean and idiomatic C++.

### 2. `mem_ref<T>` / `mem_ptr<T>` Device Memory Handles

The reference-counted GPU memory handle is a good abstraction:

- RAII semantics via `intrusive_ptr` ensure GPU memory gets freed
- Scalar vs. buffer unification keeps the API surface small
- `copy_to_host()` with stream synchronization is correct
- Tracking device ID on the handle prevents accidental cross-device use

### 3. Stream Pool & Per-Actor Stream Table

`StreamPool` and `DeviceStreamTable` are well-engineered:

- Thread-safe with appropriate lock granularity (shared lock for reads,
  exclusive for writes in `DeviceStreamTable`)
- Bounded pool with graceful degradation (round-robin reuse) rather than
  unbounded growth or crashes
- Pre-allocation of streams avoids creation latency on the hot path
- The acquire/release pattern is simple and correct

### 4. Dual API: Actor Facade + Command Runner

Offering both a high-level actor facade (send message, get reply) and a
lower-level command runner (call directly from your actor's behavior) gives users
flexibility. The actor facade is good for fire-and-forget GPU work; the command
runner is better for actors that need to orchestrate multi-step GPU pipelines.

### 5. Multi-GPU Awareness

The platform/scheduler/device hierarchy correctly handles:

- Device enumeration and per-device context creation
- Homogeneity detection (disabling multi-GPU for mixed hardware)
- Per-device kernel loading in `program::load_kernels()`
- Stream isolation per device

### 6. Control Layer Architecture

The token-based scheduling system (launch tokens, response tokens, behavior
tokens) is an ambitious and thoughtful design for GPU workload scheduling. The
concept of pluggable scheduler behaviors (green light, core usage, pressure-based)
with a behavior table for runtime switching shows real engineering ambition.

### 7. NVRTC Integration

Supporting both runtime compilation (NVRTC from source strings) and precompiled
binaries (CUBIN/fatbin) gives users deployment flexibility.

### 8. Comprehensive Test Coverage

The test suite exercises real GPU kernels (matrix multiply, vector add, string
compare), verifies results against CPU reference implementations, tests
async/sync modes, multi-actor concurrency, float/double types, and the control
layer. This is a strong foundation.

---

## The Bad

These are problems that need fixing but are tractable engineering tasks.

### 1. Global Singleton Manager Breaks CAF Isolation

**Files:** `manager.hpp`, `manager.cpp`

The `manager` uses a raw `static manager* instance_` with manual
`new`/`delete`. This is fundamentally incompatible with CAF:

- CAF supports multiple `actor_system` instances per process (essential for
  testing). A global singleton makes this impossible.
- The `shutdown()` must be called manually and is not tied to `actor_system`
  lifecycle, creating resource leak and use-after-free risks.
- The raw `new`/`delete` is an anachronism in a codebase that uses
  `intrusive_ptr` everywhere else.

### 2. Positional Integer Argument Trap in `command_runner`

**File:** `command_runner.hpp`

The `run()` and `run_async()` overloads accept `(program, dims, actor_id,
shared_mem, device_number, args...)` where `actor_id`, `shared_mem`, and
`device_number` are all `int`. Because these precede a variadic pack, omitting
one silently shifts all subsequent arguments. The compiler cannot catch this.
This is the most dangerous API design issue in the library.

### 3. Untyped Output Extraction

**Files:** `helpers.hpp`, `command.hpp`

Synchronous kernel results come back as `std::vector<output_buffer>` and must be
extracted via `extract_vector<float>(outputs, 2)`. The index is fragile: adding
a kernel parameter shifts indices with no compiler warning. Since `command_runner`
is already templated on the argument types, the output type tuple can be deduced
at compile time.

### 4. `output_buffer` and `buffer_variant` Live in Global Namespace

**File:** `types.hpp`

`buffer_variant` and `output_buffer` are defined outside any namespace. They
should be inside `caf::cuda` to avoid collisions and match the rest of the
library.

### 5. Macro-Based Access Flags

**File:** `types.hpp`

```cpp
#define IN 0
#define IN_OUT 1
#define OUT 2
#define NOT_IN_USE -1
```

Preprocessor macros for these constants pollute the global namespace and prevent
any type safety. These should be a scoped enum (`enum class access_mode`).

### 6. `nd_range` Uses `std::vector` for Fixed-Size Data

**File:** `nd_range.hpp`

Grid and block dimensions are always exactly 3 values. Using `std::vector<size_t>`
incurs heap allocation for 3 integers. An `std::array<unsigned, 3>` is the
correct choice — zero overhead, no heap, still range-checkable.

### 7. Inconsistent Error Handling Strategy

Across the codebase, errors are handled in at least four different ways:
- `CHECK_CUDA` macro throws `std::runtime_error`
- `check()` free function in `global.hpp` calls `exit(1)`
- `CHECK_NVRTC` calls `exit(1)`
- Functions like `max_active_blocks_per_sm` return 0 and print to stderr

This should be unified. CAF has `caf::error` — use it, or at minimum use
exceptions consistently (never `exit(1)` in library code).

### 8. `program` Has a Memory Leak in `intrusive_ptr_release`

**File:** `program.hpp`

```cpp
friend void intrusive_ptr_release(const program* p) noexcept {
    if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
       // delete p;  // COMMENTED OUT — known segfault
    }
}
```

The `delete` is commented out with a TODO. This means every `program` object
leaks for the lifetime of the process. The segfault is likely caused by
destruction order issues with CUDA contexts. This needs to be debugged and fixed,
not papered over.

### 9. Duplicate Reference Counting in `mem_ref`

**File:** `mem_ref.hpp`

`mem_ref` inherits from `caf::ref_counted` *and* carries its own
`mutable std::atomic<size_t> ref_count_` with custom `intrusive_ptr_add_ref` /
`intrusive_ptr_release` friends. Only one mechanism should be used. The current
setup means the base class ref count is never used, which is confusing and wastes
memory.

### 10. Thread Safety Issues in `random_number()` and Scheduling

**File:** `helpers.cpp`, `manager.cpp`, `scheduler.cpp`

- `random_number()` uses `static` thread-unsafe random state
- `manager::send_scheduler_actor_message()` uses bare `rand() % num_devices`
  which is thread-unsafe and biased
- `multi_device_scheduler::schedule()` uses `std::rand()` — same problem

All should use thread-local or properly locked random engines.

### 11. `actor_facade` Implements Its Own Mailbox

**File:** `actor_facade.hpp`

The actor facade uses a `std::queue<mailbox_element_ptr> mailbox_` and manually
implements scheduling, message dispatch, and shutdown logic. This duplicates what
CAF's `event_based_actor` and its mailbox infrastructure already do, and likely
gets subtle edge cases wrong (e.g., priority handling, flow control, monitoring).

### 12. `in_impl` Stores a Dangling Pointer on Vector Move

**File:** `types.hpp`

```cpp
explicit in_impl(std::vector<T>&& buf)
    : scalar_{}, ptr_{buf.data()}, size_{buf.size()}, is_scalar_{false} {}
```

After the constructor returns, `buf` is destroyed (it was moved in as an rvalue).
`ptr_` now points to freed memory. The vector data needs to be owned by
`in_impl`, not referenced.

### 13. Missing `#pragma once` / Include Guard Consistency

Some headers rely on `#pragma once`, others have neither pragma nor traditional
include guards. The component-actor headers need review.

### 14. CUmodule Leak in `program::load_kernels()`

**File:** `program.cpp`

`CUmodule module` is loaded per device, the `CUfunction` is extracted, but the
module handle is never stored or destroyed. According to the CUDA documentation,
modules must remain loaded for their functions to be valid. If the driver
garbage-collects the module, kernel launches will segfault.

---

## The Ugly

These are deeper architectural and code quality issues.

### 1. `check()` Free Function Calls `exit(1)` From Library Code

**File:** `global.hpp`

```cpp
void inline check(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(result, &err_str);
        std::cerr << "CUDA Driver API Error (" << msg << "): " << err_str << "\n";
        exit(1);
    }
}
```

A library must never call `exit()`. This terminates the entire process without
unwinding stacks, flushing buffers, or giving the application any chance to
handle the error. The `platform` constructor calls `check()` for every device
operation — any CUDA hiccup kills the process silently.

### 2. Entire Test Suite Lives in `src/test.cpp` Outside CAF's Test Framework

**File:** `src/test.cpp`

CAF has its own test framework (`libcaf_test`) with `TEST`, `CHECK_EQ`, etc.
The CUDA tests are plain functions called from `main()` with `std::cout`-based
assertions. This means:

- Tests don't integrate with CTest / the CI reporting
- No structured test discovery or selective execution
- Failures are reported as console output, not structured results
- Test setup/teardown doesn't use CAF's fixtures

### 3. `device.hpp` Is a 400+ Line God Header

The `device` class does too many things: memory allocation, kernel argument
preparation, kernel launching, stream management, output buffer collection,
device property caching, and legacy helpers. This should be decomposed.

### 4. Commented-Out Code and TODO Debris Throughout

Examples from the codebase:
- `//std::cout << "Hello???\n";` in `device.cpp`
- `//WARNING TURNING THIS ON FOR SOME REASON, CAUSES SEGFAUTLS` in `program.hpp`
- `//this actually doesnt even work do not use` in `manager.cpp`
- `//TODO fix issue where actors store a non in use ndrange` in `nd_range.hpp`
- `// this is legacy code, do not use` for `get_scheduler_actor()`
- Empty file: `random_mmul_gen_multiply_actor.hpp`
- `main.test.cpp` is empty

This signals incomplete work. Dead code and broken functions should be removed,
not commented.

### 5. Mixed Naming Conventions

The codebase mixes:
- `camelCase`: `getDevice()`, `getContext()`, `getStreamId()`
- `snake_case`: `get_stream_for_actor()`, `copy_to_host()`, `launch_kernel_internal()`
- `PascalCase`: `StreamPool`, `DeviceStreamTable`

CAF uses `snake_case` for functions and `snake_case` for types. The CUDA module
should follow suit.

### 6. Serialization of `std::vector<T>` in Global Namespace

**File:** `global.hpp`

Custom `inspect()` overloads for `std::vector<char>`, `std::vector<int>`,
`std::vector<float>`, `std::vector<double>` are defined in the global namespace.
These conflict with any other library (or CAF itself) that defines
serialization for these standard types.

### 7. Hard-Coded Type Support (char/int/float/double Only)

**Files:** `types.hpp`, `global.hpp`

`buffer_variant` is `std::variant<vector<char>, vector<int>, vector<float>,
vector<double>>`. Users cannot use `uint32_t`, `int64_t`, half-precision, or
custom structs without modifying the variant and adding type IDs. The type
system should be open or template-based, not closed.

### 8. `create_in_out_arg` Has Two Definitions for Different Types

**File:** `helpers.hpp`

```cpp
template <typename T>
in<T> create_in_out_arg(std::vector<T>&& buffer) {
    return in<T>{std::move(buffer)}; // BUG: returns in<T>, not in_out<T>
}

template <typename T>
in_out<T> create_in_out_arg(std::vector<T>&& buffer) {
    return in_out<T>{std::move(buffer)};
}
```

The first definition returns `in<T>` instead of `in_out<T>`, which is either a
copy-paste bug or will cause an ODR violation / ambiguous overload.

---

## What to Keep

1. **The `in<T>` / `out<T>` / `in_out<T>` tagged type system** — This is the
   identity of the library. Refine it (fix the dangling pointer, add type safety)
   but keep the concept.

2. **`mem_ref<T>` / `mem_ptr<T>`** — Reference-counted GPU memory handles are
   correct. Fix the dual ref-counting, but keep the abstraction.

3. **`StreamPool` and `DeviceStreamTable`** — Well-designed, keep as-is with
   minor naming fixes.

4. **The dual API (actor facade + command runner)** — Both usage patterns are
   valuable. The actor facade needs a rewrite to use CAF's actor infrastructure
   properly, but the concept is right.

5. **Multi-GPU support in `platform` / `scheduler`** — The device enumeration,
   homogeneity detection, and scheduling abstraction are good foundations.

6. **The control-layer token/behavior system** — Ambitious and forward-thinking.
   Needs cleanup but the design intent is sound.

7. **NVRTC + precompiled binary support** — Keep both paths.

8. **The existing test scenarios** — The matrix multiply, vector add, and
   concurrency tests are valuable. Port them to CAF's test framework.

## What to Change

1. **Global singleton manager** → proper CAF module
2. **Positional int args** → `launch_config` struct
3. **Untyped output** → compile-time deduced output tuples
4. **`exit()` calls in library code** → exceptions or `caf::error`
5. **`actor_facade` custom mailbox** → inherit from `event_based_actor`
6. **Dangling pointer in `in_impl` rvalue ctor** → own the data
7. **Macro access flags** → `enum class`
8. **`nd_range` heap alloc** → `std::array`
9. **`program` memory leak** → fix destruction order, store `CUmodule`
10. **Thread-unsafe RNG** → thread-local engines
11. **Mixed naming** → consistent `snake_case` per CAF convention
12. **Global namespace types** → move into `caf::cuda`
13. **Hard-coded variant types** → template-based or extensible registration
14. **Test infrastructure** → port to `libcaf_test` + CTest

---

## Roadmap

### Phase 1 — Safety & Correctness

*Goal: Eliminate bugs that cause crashes, data corruption, or undefined behavior.*

| # | Task | Priority | Files |
|---|------|----------|-------|
| 1.1 | **Fix dangling pointer in `in_impl` rvalue constructor** — `in_impl(vector<T>&&)` stores `buf.data()` but `buf` is destroyed after construction. Make `in_impl` own the data (store a `vector<T>` member when constructed from rvalue). Same issue exists in `in_out_impl`. | Critical | `types.hpp` |
| 1.2 | **Fix `program` memory leak** — Uncomment `delete p` in `intrusive_ptr_release`. Debug the segfault (likely caused by `CUmodule` being destroyed before `CUfunction` is used, or context ordering). Store `CUmodule` handles in the `program` object and destroy them in the destructor after pushing the correct context. | Critical | `program.hpp`, `program.cpp` |
| 1.3 | **Fix duplicate ref-counting in `mem_ref`** — Remove the custom `atomic<size_t> ref_count_` and custom `intrusive_ptr_add_ref`/`intrusive_ptr_release`. Use only the base `caf::ref_counted` mechanism. | Critical | `mem_ref.hpp` |
| 1.4 | **Eliminate `exit(1)` calls in library code** — Replace `check()` function and `CHECK_NVRTC` macro with exception throws. Library code must never terminate the process. | Critical | `global.hpp` |
| 1.5 | **Fix `create_in_out_arg` returning wrong type** — The overload taking `vector<T>&&` returns `in<T>` instead of `in_out<T>`. Fix to return `in_out<T>`. | High | `helpers.hpp` |
| 1.6 | **Fix thread-unsafe random number generation** — Make `random_number()` use `thread_local` storage. Replace `rand()` calls in `manager.cpp` and `scheduler.cpp` with the thread-safe version. | High | `helpers.cpp`, `manager.cpp`, `scheduler.cpp` |
| 1.7 | **Store `CUmodule` handles in `program`** — Currently `CUmodule` is a local variable in `load_kernels()` and is never stored. The CUDA driver may unload the module, invalidating the `CUfunction`. Store modules in a `map<int, CUmodule>` and destroy them in the `program` destructor. | High | `program.hpp`, `program.cpp` |
| 1.8 | **Replace `#define IN/OUT/IN_OUT` with `enum class access_mode`** — Preprocessor macros for constants pollute the global namespace. Use `enum class access_mode { in, in_out, out, not_in_use }`. | Medium | `types.hpp`, all files using `IN`/`OUT`/`IN_OUT` |
| 1.9 | **Move `buffer_variant` and `output_buffer` into `caf::cuda` namespace** | Medium | `types.hpp`, `global.hpp` |
| 1.10 | **Remove global-namespace `inspect()` overloads for `std::vector<T>`** — These conflict with CAF's own serialization. Only provide inspect for CUDA-specific types. | Medium | `global.hpp` |

### Phase 2 — CAF Integration

*Goal: Make libcaf_cuda a first-class CAF module that follows the framework's
conventions and lifecycle.*

| # | Task | Priority | Files |
|---|------|----------|-------|
| 2.1 | **Convert `manager` to a proper CAF module** — Create `caf::cuda::middleman` inheriting from `actor_system_module`. Register via `CAF_MAIN(caf::cuda::middleman)`. Module owns platform, devices, programs. `init()` replaces `manager::init()`. `stop()` replaces `manager::shutdown()`. Provide `sys.get_module<caf::cuda::middleman>()` accessor. | Critical | New `middleman.hpp/cpp`, refactor `manager.hpp/cpp` |
| 2.2 | **Rewrite `actor_facade` to extend `event_based_actor`** — Remove the custom mailbox, custom `resume()`, custom `enqueue()`, and custom `launch()`. Use CAF's built-in mailbox and scheduling. Define a behavior in `make_behavior()` that handles kernel arguments and replies with results. | High | `actor_facade.hpp` |
| 2.3 | **Integrate type IDs via CMake `ENUM_TYPES`** — Currently type IDs are registered in `global.hpp` via manual `CAF_BEGIN_TYPE_ID_BLOCK`. Use the standard CMake `caf_add_component(ENUM_TYPES ...)` pattern like `libcaf_io` and `libcaf_net`. | High | `CMakeLists.txt`, `global.hpp` |
| 2.4 | **Port tests to CAF's test framework** — Move test scenarios from `src/test.cpp` and the `tests/` directory into proper `TEST("name")` blocks using `libcaf_test`. Register them in `CMakeLists.txt` so they appear in CTest. | High | `CMakeLists.txt`, all test files |
| 2.5 | **Unify error handling with `caf::error`** — Define a CUDA error category. Convert CUDA/NVRTC error codes to `caf::error` values that can flow through CAF's error propagation (response promises, error handlers). | Medium | New `error.hpp`, all error sites |
| 2.6 | **Apply consistent `snake_case` naming** — Rename `getDevice()` → `get_device()`, `getContext()` → `get_context()`, `StreamPool` → `stream_pool`, `DeviceStreamTable` → `device_stream_table`, etc. to match CAF conventions. | Medium | All headers and sources |

### Phase 3 — API Ergonomics

*Goal: Make the library pleasant and safe to use.*

| # | Task | Priority | Files |
|---|------|----------|-------|
| 3.1 | **Introduce `launch_config` struct** — Replace positional `(actor_id, shared_mem, device_number)` integers with a named configuration struct: `struct launch_config { nd_range dims; int actor_id; int device = -1; int shared_mem = 0; };`. Update `command_runner::run` and `run_async` signatures to `(program_ptr, launch_config, Ts...)`. | Critical | New `launch_config.hpp`, `command_runner.hpp`, `command.hpp`, `base_command.hpp` |
| 3.2 | **Type-safe output tuples** — Use template metaprogramming to filter `out<T>` and `in_out<T>` from the `command_runner` template parameters. `run()` returns `std::tuple<std::vector<raw_t<OutTs>>...>` for the output/inout types only. `run_async()` returns `std::tuple<mem_ptr<raw_t<OutTs>>...>`. Deprecate `extract_vector`. | High | `command_runner.hpp`, `command.hpp` |
| 3.3 | **Replace `nd_range` internals with `std::array`** — Change `dim_vec gridDim` and `dim_vec blockDim` from `std::vector<size_t>` to `std::array<unsigned, 3>`. Remove the heap allocation. Keep the same public API. | Medium | `nd_range.hpp` |
| 3.4 | **Extend type support beyond char/int/float/double** — Make `buffer_variant` / `output_buffer` extensible or remove them entirely in favor of the type-safe tuple returns from 3.2. If they remain for the actor facade path, allow user registration of additional types. | Medium | `types.hpp`, `global.hpp` |
| 3.5 | **Clean up dead code** — Remove: empty `random_mmul_gen_multiply_actor.hpp`, broken `create_program_from_ptx`, `get_scheduler_actor()` legacy method, commented-out debug prints, empty `main.test.cpp`. | Low | Multiple files |
| 3.6 | **Add documentation** — Create a user guide in `manual/cuda/` covering: module setup, spawning GPU actors, using `command_runner`, memory management with `mem_ptr`, multi-GPU configuration, and the control-layer scheduler. | Low | New `manual/cuda/` |

### Phase 4 — Runtime API Migration

*Goal: Support modern CUDA C++ workflows where host and device code coexist.*

| # | Task | Priority | Files |
|---|------|----------|-------|
| 4.1 | **Add CMake `enable_language(CUDA)` support** — Allow `.cu` files to be compiled directly as part of the build, alongside the Driver API path. This is additive, not a replacement. | High | `CMakeLists.txt` |
| 4.2 | **Add Runtime API kernel launch path** — Create a `runtime_command` that wraps `cudaLaunchKernel` and accepts a function pointer to a `__global__` function instead of a `program_ptr`. | High | New `runtime_command.hpp` |
| 4.3 | **Unified kernel handle** — Abstract `program_ptr` (Driver API) and function pointer (Runtime API) behind a common `kernel_handle` type so `command_runner` and actor facade work with both. | Medium | New `kernel_handle.hpp` |
| 4.4 | **Evaluate removing NVRTC path** — Once Runtime API is available, NVRTC runtime compilation may be unnecessary for most users. Consider making it an optional feature behind a CMake flag. | Low | `helpers.hpp/cpp`, `manager.cpp` |

---

## Actionable Task Descriptions

### Task 1.1: Fix Dangling Pointer in Wrapper Types

**Problem:** `in_impl(std::vector<T>&& buf)` stores `buf.data()` in `ptr_`, but
`buf` is an rvalue that gets destroyed when the constructor exits. `ptr_` points
to freed memory.

**Fix:** Add an `std::vector<T> owned_data_` member. When constructed from an
rvalue vector, move the data into `owned_data_` and point `ptr_` at
`owned_data_.data()`. When constructed from an lvalue reference, keep the current
non-owning pointer behavior. Apply the same fix to `in_out_impl`.

**Verification:** Write a test that passes `std::vector<float>{1,2,3}` as a
temporary to `in<float>` and reads back via `data()`.

---

### Task 1.2: Fix Program Memory Leak

**Problem:** `intrusive_ptr_release` for `program` has `delete p` commented out
due to segfaults.

**Root cause investigation:**
1. `program::load_kernels()` creates `CUmodule` as a local variable — it goes
   out of scope but `CUfunction` still references it. If the driver unloads the
   module, the function handle becomes invalid.
2. The `program` destructor (implicitly) does nothing with CUDA resources, so
   there's no proper cleanup ordering.

**Fix:**
1. Store `std::unordered_map<int, CUmodule> modules_` in `program`.
2. In the destructor, push each device's context, call `cuModuleUnload`, pop.
3. Uncomment `delete p` in `intrusive_ptr_release`.
4. Alternatively, inherit properly from `ref_counted` and remove custom ref
   counting (like task 1.3 for `mem_ref`).

---

### Task 2.1: Convert to CAF Module

**Problem:** `manager` is a global singleton that doesn't participate in CAF's
module lifecycle.

**Design:**

```cpp
namespace caf::cuda {

class middleman : public actor_system_module {
public:
  static void init_global_meta_objects();
  static actor_system_module* make(actor_system& sys);
  static void add_module_options(actor_system_config& cfg);
  
  // actor_system_module interface
  void start() override;
  void stop() override;
  void init(actor_system_config&) override;
  id_t id() const override;
  void* subtype_ptr() override;

  // Public API (replaces manager::get())
  program_ptr create_program(const char* source, const std::string& name);
  program_ptr load_program_from_cubin(const std::string& path, const char* name);
  device_ptr find_device(int id);
  int num_devices();
  
private:
  actor_system& system_;
  platform_ptr platform_;
  std::vector<actor> scheduler_actors_;
  actor memory_actor_;
};

} // namespace caf::cuda
```

**Usage:**
```cpp
// In main
void caf_main(actor_system& sys) {
  auto& cuda = sys.get_module<caf::cuda::middleman>();
  auto prog = cuda.load_program_from_cubin("kernel.cubin", "matMul");
  // ...
}
CAF_MAIN(caf::cuda::middleman)
```

---

### Task 3.1: Introduce `launch_config`

**Problem:** `run(program, dims, actor_id, shared_mem, device_number, args...)`
is a positional integer minefield.

**Design:**

```cpp
namespace caf::cuda {

struct launch_config {
  nd_range dims;
  int actor_id;
  int device = -1;       // -1 = auto-schedule
  int shared_mem = 0;    // bytes
};

template <class... Ts>
class command_runner {
public:
  // Clean signature — config is a single named struct
  template <class... Us>
  auto run(program_ptr program, launch_config config, Us&&... xs);

  template <class... Us>
  auto run_async(program_ptr program, launch_config config, Us&&... xs);
};

} // namespace caf::cuda
```

Keep the old overloads temporarily but mark them `[[deprecated]]`.

---

### Task 3.2: Type-Safe Output Tuples

**Design sketch:**

```cpp
// Filter: keep only out<T> and in_out<T> from a type list
template <class... Ts>
struct output_types;

// For run(): return std::tuple<std::vector<raw_t<OutT>>...>
// For run_async(): return std::tuple<mem_ptr<raw_t<OutT>>...>

template <class... Ts>
class command_runner {
  using out_tuple_t = /* filtered tuple of raw output types */;
  
  template <class... Us>
  std::tuple<std::vector<raw_t<out_types>>...>
  run(program_ptr prog, launch_config cfg, Us&&... xs);
};

// Usage:
// command_runner<in<float>, in<float>, out<float>> mmul;
// auto [result] = mmul.run(prog, cfg, matA, matB, out<float>(N*N));
```

---

## Dependency Graph

```
Phase 1 (Safety)         Phase 2 (CAF)           Phase 3 (API)         Phase 4 (Runtime)
─────────────────        ─────────────           ─────────────         ─────────────────
1.1 Fix dangling ptr ──┐
1.2 Fix program leak ──┤
1.3 Fix dual refcount ─┤
1.4 Remove exit() ─────┼── 2.1 CAF module ─────── 3.1 launch_config
1.5 Fix create_in_out ─┤   2.2 Rewrite facade     3.2 Typed output ──── 4.1 CMake CUDA
1.6 Thread-safe RNG ───┤   2.3 CMake type IDs     3.3 nd_range array    4.2 Runtime command
1.7 Store CUmodule ────┤   2.4 Port tests         3.4 Extend types      4.3 Unified handle
1.8 enum class access ─┤   2.5 caf::error         3.5 Clean dead code   4.4 Optional NVRTC
1.9 Namespace types ───┘   2.6 Naming             3.6 Documentation
1.10 Remove inspect
```

Phase 1 has no internal dependencies and all tasks can be done in parallel.
Phase 2 depends on Phase 1 completion (especially 1.4 and 1.8).
Phase 3 depends on Phase 2 (especially 2.1 for the module accessor).
Phase 4 can start in parallel with Phase 3 at the CMake level.
