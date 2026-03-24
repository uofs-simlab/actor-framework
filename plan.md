# libcaf_cuda Plan

## Goal

Move `libcaf_cuda` from a working prototype to a CAF-aligned asynchronous GPU integration where:

1. actors submit GPU work without blocking CAF workers
2. GPU completion is surfaced back into CAF as an event
3. memory can remain on the GPU until an actor explicitly decides to copy it back
4. `actor_facade` and `command_runner` become two APIs over one shared runtime

## Phase 1: Fix the Hard Blockers

Priority: `P0 Critical`

### 1. Build a true async execution path

Outcome:

- GPU submission returns immediately to CAF
- completion arrives later as a CAF message or promise fulfillment

Work:

- add a submission path that launches kernels without collecting outputs immediately
- capture `response_promise` in `actor_facade` instead of returning outputs from the same handler
- introduce a completion notification mechanism
  - preferred options to evaluate:
    - CUDA event plus a detached completion watcher
    - CUDA event plus polling coordinator actor
    - host callback only if it is safe with the chosen driver/runtime setup
- store pending request state keyed by request ID or launch handle

Files likely touched:

- [libcaf_cuda/caf/cuda/actor_facade_event.hpp](libcaf_cuda/caf/cuda/actor_facade_event.hpp)
- [libcaf_cuda/caf/cuda/command.hpp](libcaf_cuda/caf/cuda/command.hpp)
- [libcaf_cuda/caf/cuda/command_runner.hpp](libcaf_cuda/caf/cuda/command_runner.hpp)
- [libcaf_cuda/caf/cuda/device.hpp](libcaf_cuda/caf/cuda/device.hpp)
- [libcaf_cuda/caf/cuda/mem_ref.hpp](libcaf_cuda/caf/cuda/mem_ref.hpp)

### 2. Separate launch completion from host copy-back

Outcome:

- actors can choose whether to keep memory on device or copy later

Work:

- define a result type for completed device work that contains `mem_ptr`s or a launch handle
- add explicit copy-back helpers outside the immediate launch path
- redesign facade policy:
  - default facade may still offer convenience copy-back
  - but underlying runtime must not force it

### 3. Fix `program` ownership and deletion

Outcome:

- no leaked `program`
- no segfault on teardown

Work:

- determine ownership of `CUmodule` and `CUfunction`
- store and unload `CUmodule` handles explicitly
- audit interaction with context destruction order
- add tests for create/destroy cycles

Files likely touched:

- [libcaf_cuda/caf/cuda/program.hpp](libcaf_cuda/caf/cuda/program.hpp)
- [libcaf_cuda/src/program.cpp](libcaf_cuda/src/program.cpp)
- [libcaf_cuda/caf/cuda/platform.hpp](libcaf_cuda/caf/cuda/platform.hpp)
- [libcaf_cuda/caf/cuda/device.hpp](libcaf_cuda/caf/cuda/device.hpp)

### 4. Remove prototype residue from runtime code

Outcome:

- no test stubs in production control paths

Work:

- replace the `"Hello world from memory response"` release path with the real protocol or remove the unused type entirely
- remove or quarantine broken public APIs marked "do not use"

## Phase 2: Unify the Runtime Model

Priority: `P1 High`

### 5. Define one execution engine beneath both APIs

Outcome:

- `actor_facade` and `command_runner` are distinct interfaces, not distinct runtime models

Work:

- introduce a shared execution object or service for:
  - launch submission
  - stream/device selection
  - completion tracking
  - memory retention
  - host-copy operations
- make `command_runner` the low-level interface over that engine
- make `actor_facade` a CAF wrapper over that same engine

### 6. Clarify stream semantics and exhaustion behavior

Outcome:

- stream reuse is safe and documented

Work:

- decide whether stream exhaustion should:
  - block
  - allocate more
  - fail fast
  - reuse only after explicit completion guarantees
- stop silent round-robin reuse if it can break correctness
- add observability for stream pressure

### 7. Replace global `platform` singleton

Outcome:

- runtime state belongs to the `manager` / `actor_system`

Work:

- move platform lifetime under `manager`
- make contexts, devices, schedulers, and pools actor-system scoped
- ensure shutdown order is deterministic

## Phase 3: Simplify and Align with CAF

Priority: `P2 Medium`

### 8. Reassess the control layer

Outcome:

- either a justified subsystem or a smaller, clearer one

Work:

- decide which control-layer parts are essential
- remove or isolate incomplete scheduler behaviors
- simplify token traffic if standard CAF request/reply and scheduling are sufficient

### 9. Improve message safety and distribution boundaries

Outcome:

- clearer local-only versus serializable API boundaries

Work:

- document which GPU objects are intentionally local-only
- reduce `CAF_ALLOW_UNSAFE_MESSAGE_TYPE` where possible
- avoid exposing unsafe types through broad public headers unless necessary

### 10. Normalize error behavior

Outcome:

- consistent caller expectations

Work:

- define where errors are thrown versus returned as `caf::error`
- align facade, command runner, manager, and control-layer error paths

## Phase 4: Harden the Module

Priority: `P3 Lower`

### 11. Expand focused tests

Outcome:

- regression protection around the new async model

Add tests for:

- facade does not block worker thread during GPU execution
- completion notification arrives after kernel completion
- memory can remain on GPU across multiple launches
- explicit host copy works after delayed completion
- teardown of programs, contexts, and streams is clean
- stream exhaustion policy behaves as designed

### 12. Refresh documentation

Outcome:

- docs match actual semantics

Work:

- update `README.md` and `documentation.txt`
- clearly distinguish:
  - sync convenience behavior
  - async execution model
  - local-only GPU handles
  - actor facade versus command runner roles

## Recommended Order of Execution

1. Fix `program` lifetime and teardown bugs.
2. Implement asynchronous launch completion without forced copy-back.
3. Redesign `actor_facade` around `response_promise` and completion events.
4. Rebase `command_runner` on the same async engine.
5. Make stream reuse and device ownership safe and explicit.
6. Clean up control-layer prototype residue.
7. Harden tests and documentation.

## Concrete Next Slice

If you want to start implementation immediately, the best next slice is:

1. add a launch handle or completion token that represents submitted GPU work
2. add an async facade path that stores `response_promise` and returns immediately
3. add explicit `copy_to_host` only after completion
4. verify with one end-to-end matrix multiply example

That slice directly attacks the highest-value architectural gap without requiring the full control-layer redesign first.