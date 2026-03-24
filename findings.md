# libcaf_cuda Findings

## Executive Summary

`libcaf_cuda` is a credible working prototype for integrating CUDA execution into CAF, but it is still much closer to a research branch than a production-quality CAF module.

The good news is that the foundation is real:

- there is a usable module entry point through `manager`
- there is a consistent kernel launch core built around `program`, `device`, `command`, `mem_ref`, and `nd_range`
- both `actor_facade` and `command_runner` already work on top of that same core
- examples and tests are broad enough to show the intended programming model

The central problem is that the current design does **not yet achieve true asynchronous CAF integration**. In the main user-facing path, CPU-side CAF workers still block while GPU work is finishing and while data is copied back to host. There is also significant architectural drift from CAF conventions: global singletons, unsafe message types, optional side subsystems, and several unfinished or explicitly broken code paths.

My conclusion is:

- `actor_facade` and `command_runner` should both continue to exist
- but they should exist as **two APIs over one unified async execution engine**, not as separate conceptual implementations
- the first major engineering goal should be to separate **submission**, **completion**, and **host-copy timing**

## Architecture Overview

### Public API Layer

- [libcaf_cuda/caf/cuda/all.hpp](libcaf_cuda/caf/cuda/all.hpp)
  - umbrella include for the CUDA module
- [libcaf_cuda/caf/cuda/manager.hpp](libcaf_cuda/caf/cuda/manager.hpp)
  - CAF module entry point, device access, program construction, facade spawning
- [libcaf_cuda/caf/cuda/actor_facade.hpp](libcaf_cuda/caf/cuda/actor_facade.hpp)
- [libcaf_cuda/caf/cuda/actor_facade_event.hpp](libcaf_cuda/caf/cuda/actor_facade_event.hpp)
  - actor-oriented GPU gateway
- [libcaf_cuda/caf/cuda/command_runner.hpp](libcaf_cuda/caf/cuda/command_runner.hpp)
  - direct command-oriented GPU interface for custom actors and pipelines

### Compute and Resource Core

- [libcaf_cuda/caf/cuda/program.hpp](libcaf_cuda/caf/cuda/program.hpp)
- [libcaf_cuda/src/program.cpp](libcaf_cuda/src/program.cpp)
  - wraps a kernel binary and resolves device-specific `CUfunction` handles
- [libcaf_cuda/caf/cuda/device.hpp](libcaf_cuda/caf/cuda/device.hpp)
- [libcaf_cuda/src/device.cpp](libcaf_cuda/src/device.cpp)
  - owns device context, stream table, occupancy queries, memory allocation and launch helpers
- [libcaf_cuda/caf/cuda/platform.hpp](libcaf_cuda/caf/cuda/platform.hpp)
- [libcaf_cuda/src/platform.cpp](libcaf_cuda/src/platform.cpp)
  - global device registry and scheduler selection point
- [libcaf_cuda/caf/cuda/command.hpp](libcaf_cuda/caf/cuda/command.hpp)
  - common kernel launch abstraction
- [libcaf_cuda/caf/cuda/mem_ref.hpp](libcaf_cuda/caf/cuda/mem_ref.hpp)
  - intrusive GPU memory handle used across sync and async paths
- [libcaf_cuda/caf/cuda/streampool.hpp](libcaf_cuda/caf/cuda/streampool.hpp)
- [libcaf_cuda/src/streampool.cpp](libcaf_cuda/src/streampool.cpp)
  - per-device stream pooling and actor-to-stream mapping

### Control Layer

- [libcaf_cuda/caf/cuda/control-layer/all-control-layer.hpp](libcaf_cuda/caf/cuda/control-layer/all-control-layer.hpp)
- [libcaf_cuda/caf/cuda/control-layer/scheduler_actor.hpp](libcaf_cuda/caf/cuda/control-layer/scheduler_actor.hpp)
- [libcaf_cuda/src/control-layer/scheduler_actor.cpp](libcaf_cuda/src/control-layer/scheduler_actor.cpp)
- [libcaf_cuda/src/control-layer/memory_actor/memory_actor.cpp](libcaf_cuda/src/control-layer/memory_actor/memory_actor.cpp)
- [libcaf_cuda/caf/cuda/control-layer/behavior_table.hpp](libcaf_cuda/caf/cuda/control-layer/behavior_table.hpp)

This is a second subsystem layered on top of the compute core. It introduces scheduler actors, tokens, reclaim messages, memory grants, and policy behaviors.

It is more experimental than the core execution layer.

## How Well It Fits CAF

### Good Fit

1. It uses a real CAF module entry point.
   - `manager` derives from `actor_system_module`, which is the right extension point for runtime integration.

2. The facade can now be expressed as an `event_based_actor`.
   - This is much closer to CAF's actual actor model than the old custom mailbox implementation.

3. The message-based GPU API is conceptually aligned with CAF.
   - Spawn actor, send request, receive response is a good fit for actor users.

4. The command runner gives advanced users a lower-level path without forcing everything through one actor abstraction.

### Weak Fit

1. Core GPU objects are still effectively local-only and outside normal CAF distribution semantics.
   - [libcaf_cuda/caf/cuda/global.hpp](libcaf_cuda/caf/cuda/global.hpp)
   - [libcaf_cuda/caf/cuda/control-layer/all-control-layer.hpp](libcaf_cuda/caf/cuda/control-layer/all-control-layer.hpp)
   - both rely heavily on `CAF_ALLOW_UNSAFE_MESSAGE_TYPE`

2. The main execution model is not scheduler-friendly.
   - The facade returns replies through CAF correctly, but it still blocks while waiting for GPU completion.

3. The `platform` singleton is not aligned with multi-`actor_system` process structure.
   - [libcaf_cuda/src/platform.cpp](libcaf_cuda/src/platform.cpp)

4. The control layer behaves like a custom side-runtime inside CAF instead of a well-integrated CAF service.

## The Good

### 1. There is already a clean shared compute core

Both user-facing APIs ultimately converge on the same lower-level launch machinery:

- `program`
- `device`
- `command` / `base_command`
- `mem_ref`
- `nd_range`

That is a strong architectural asset because it means the system is not fundamentally split in two.

### 2. The user-facing facade API is strong conceptually

The facade gives the right shape for CAF users:

- spawn a GPU-backed actor
- send tagged kernel arguments
- receive ordinary CAF replies

That is exactly the right direction for making GPU work feel native in actor code.

### 3. The command runner is valuable for advanced control

`command_runner` supports:

- sync launches
- async launches returning device memory handles
- manual memory transfer
- explicit device selection
- shared memory size
- stream selection via actor ID or token

That is useful and should remain available for sophisticated pipelines and component actors.

### 4. Per-actor stream assignment is a sensible first model

The stream table gives a useful invariant:

- the same actor ID gets the same stream until released

That helps preserve ordering and keeps GPU-side execution understandable.

### 5. The prototype has real breadth

There are examples, actor-facade tests, command-runner tests, component actor tests, benchmark tests, and control-layer tests. That is a much better starting point than a minimal demo.

## The Bad

### 1. The facade is still synchronous in the critical path

Priority: `P0 Critical`

Relevant files:

- [libcaf_cuda/caf/cuda/actor_facade_event.hpp](libcaf_cuda/caf/cuda/actor_facade_event.hpp)
- [libcaf_cuda/caf/cuda/command.hpp](libcaf_cuda/caf/cuda/command.hpp)
- [libcaf_cuda/caf/cuda/mem_ref.hpp](libcaf_cuda/caf/cuda/mem_ref.hpp)

The facade launches GPU work asynchronously onto a stream, but then immediately collects outputs and synchronizes in `copy_to_host()`.

Effect:

- a CAF scheduler thread remains occupied while waiting on GPU completion
- this undermines actor fairness and scheduler scalability
- the implementation does not yet meet the intended asynchronous actor model

### 2. The API split is useful, but the implementation story is not unified enough

Priority: `P1 High`

Relevant files:

- [libcaf_cuda/caf/cuda/actor_facade_event.hpp](libcaf_cuda/caf/cuda/actor_facade_event.hpp)
- [libcaf_cuda/caf/cuda/command_runner.hpp](libcaf_cuda/caf/cuda/command_runner.hpp)

`actor_facade` and `command_runner` are mostly two entry points into the same execution core, but they present different lifecycle and memory semantics without a clearly documented common model.

Current state:

- facade implies automatic copy-back and reply
- command runner exposes manual memory retention and explicit host copy

That is fine as an API distinction, but the code should make it explicit that they are layered over one unified execution engine.

### 3. Global singleton platform is a bad fit for CAF modularity

Priority: `P1 High`

Relevant files:

- [libcaf_cuda/caf/cuda/platform.hpp](libcaf_cuda/caf/cuda/platform.hpp)
- [libcaf_cuda/src/platform.cpp](libcaf_cuda/src/platform.cpp)

`platform::create()` returns a process-global singleton and creates CUDA contexts globally.

Problems:

- weak fit for multiple `actor_system` instances in one process
- difficult teardown semantics
- hidden global state makes tests and embedding harder
- lifetime coupling between `manager`, `program`, `device`, and CUDA contexts becomes fragile

### 4. Unsafe message types are used extensively

Priority: `P1 High`

Relevant files:

- [libcaf_cuda/caf/cuda/global.hpp](libcaf_cuda/caf/cuda/global.hpp)
- [libcaf_cuda/caf/cuda/control-layer/all-control-layer.hpp](libcaf_cuda/caf/cuda/control-layer/all-control-layer.hpp)

This is acceptable for a local prototype, but it means:

- no robust serialization story for many important GPU-related objects
- weak fit with distributed CAF
- high risk of accidental misuse across boundaries

### 5. The control layer is too experimental for the core critical path

Priority: `P2 Medium`

Relevant files:

- [libcaf_cuda/src/control-layer/scheduler_actor.cpp](libcaf_cuda/src/control-layer/scheduler_actor.cpp)
- [libcaf_cuda/caf/cuda/control-layer/behavior_table.hpp](libcaf_cuda/caf/cuda/control-layer/behavior_table.hpp)
- [libcaf_cuda/src/control-layer/memory_actor/memory_actor.cpp](libcaf_cuda/src/control-layer/memory_actor/memory_actor.cpp)

There are multiple policy behaviors, token types, reclaim pathways, and actor-based schedulers, but several parts are incomplete or weakly documented.

This makes the subsystem hard to reason about and hard to align with a clear async design.

## The Ugly

### 1. `program` currently does not delete itself because enabling deletion causes segfaults

Priority: `P0 Critical`

Relevant file:

- [libcaf_cuda/caf/cuda/program.hpp](libcaf_cuda/caf/cuda/program.hpp)

This is the strongest red-flag in the codebase.

The current code comments explicitly say that deleting `program` causes segfaults, and the actual `delete` is commented out.

Implications:

- definite leak
- unresolved lifetime bug
- likely ownership mismatch between kernel handles, modules, contexts, or device lifetimes

### 2. `memory_response_token` still sends a test string in `release()`

Priority: `P0 Critical`

Relevant file:

- [libcaf_cuda/caf/cuda/control-layer/memory_response_token.hpp](libcaf_cuda/caf/cuda/control-layer/memory_response_token.hpp)

This is prototype residue in production code. The destructor-triggered release path still sends:

- `"Hello world from memory response"`

instead of an actual memory accounting message.

That means this path is either unfinished, misleading, or unused in real workflows.

### 3. Explicitly broken APIs are part of the public surface

Priority: `P1 High`

Relevant file:

- [libcaf_cuda/caf/cuda/manager.hpp](libcaf_cuda/caf/cuda/manager.hpp)
- [libcaf_cuda/src/manager.cpp](libcaf_cuda/src/manager.cpp)

The public API still exposes methods labeled:

- "Currently not working DO NOT USE"
- "Currently broken DO not use"

That is acceptable during prototyping but should not remain in a module that aims to align with CAF.

### 4. Stream pool exhaustion intentionally falls back to reusing in-flight streams

Priority: `P1 High`

Relevant files:

- [libcaf_cuda/caf/cuda/streampool.hpp](libcaf_cuda/caf/cuda/streampool.hpp)
- [libcaf_cuda/src/streampool.cpp](libcaf_cuda/src/streampool.cpp)

When the pool is exhausted, it reuses streams round-robin instead of failing.

This is practical, but it is dangerous because reuse can collide with in-flight work and silently break ordering assumptions if callers rely on stream uniqueness under concurrency.

### 5. Resource teardown and context ownership remain unclear

Priority: `P1 High`

Relevant files:

- [libcaf_cuda/src/platform.cpp](libcaf_cuda/src/platform.cpp)
- [libcaf_cuda/caf/cuda/device.hpp](libcaf_cuda/caf/cuda/device.hpp)
- [libcaf_cuda/caf/cuda/program.hpp](libcaf_cuda/caf/cuda/program.hpp)

`device` destroys contexts in its destructor, `platform` owns devices globally, and `program` loads per-device module state but does not seem to own/unload modules explicitly.

This is probably connected to the `program` deletion bug.

## actor_facade and command_runner

## What Is the Same

1. Both use the same core launch abstractions.
   - both converge on `command` / `base_command`
   - both ultimately call into `device`

2. Both use the same wrapper/tagging model.
   - `in<T>`, `out<T>`, `in_out<T>`

3. Both rely on the same program representation.
   - `program_ptr`

4. Both use actor IDs as stream selection keys.

5. Both can launch the same kernels with the same `nd_range`.

## What Is Different

### actor_facade

- actor-centric API
- request/reply interface through CAF messaging
- automatic host-side output collection
- automatic reply delivery
- easier for normal CAF users
- currently synchronous from the scheduler's perspective because it collects outputs before returning

### command_runner

- function-call API, not an actor abstraction
- intended for custom GPU actors or manual orchestration
- exposes both `run()` and `run_async()`
- can keep memory on the GPU via `mem_ptr`
- supports explicit memory transfer and more advanced control patterns
- does not itself handle CAF request/reply semantics

## Should Both Exist?

Yes, but not as competing implementations.

Recommended model:

- `command_runner` should be the lower-level execution and memory API
- `actor_facade` should be a higher-level actor wrapper built on top of the same async engine

So the answer is not "pick one".

The answer is:

- keep both
- reduce duplication
- define one shared async runtime path
- make facade a policy/API layer rather than a separate execution concept

## Priority-Ranked Improvements

### P0 Critical

1. Make GPU submission/completion truly asynchronous for CAF workers.
2. Fix `program` lifetime and deletion semantics.
3. Remove test or placeholder behavior from `memory_response_token`.
4. Separate GPU completion from automatic host copy so actors can decide when to copy back.

### P1 High

1. Replace the `platform` singleton with `manager`-owned runtime state.
2. Unify `actor_facade` and `command_runner` around one async execution engine.
3. Clarify stream reuse policy and make exhaustion behavior explicit and safe.
4. Remove or quarantine broken public APIs such as PTX support until they work.
5. Define a cleaner ownership model for CUDA contexts, modules, streams, and memory.

### P2 Medium

1. Simplify or isolate the control-layer scheduler subsystem.
2. Improve serialization boundaries and reduce reliance on unsafe message types.
3. Normalize error handling across facade, command runner, manager, and control-layer paths.
4. Improve module-level documentation so intended semantics match actual behavior.

### P3 Lower

1. Consolidate examples and documentation around the intended modern path.
2. Trim legacy and "do not use" public entry points.
3. Add architectural tests specifically for async completion, memory retention, and scheduler fairness.

## Bottom Line

`libcaf_cuda` already proves that GPU work can be expressed naturally in CAF, but the current implementation still behaves like a synchronous prototype wrapped in actor APIs.

The right next step is not more surface API work.

The right next step is to turn the shared execution core into a **real async GPU runtime for CAF**, then let both the facade and command runner ride on top of it.