# CUDA Control Layer Walkthrough

## Big picture

The CUDA integration has two overlapping layers:

1. A direct execution layer built around `manager`, `platform`, `device`, and `command_runner`.
2. A control layer built around scheduler actors, memory actors, and token objects.

The direct layer is where kernels are compiled, device buffers are created, and `cuLaunchKernel` is ultimately called.

The control layer is where the project tries to coordinate GPU work using CAF actors and token passing.

These two layers are connected, but they are not the same thing and they are not used uniformly by every API surface.

## The main pieces

### `manager`

`manager` is the CAF module entry point for CUDA support.

Responsibilities:

- initializes CUDA via `cuInit`
- creates the `platform`
- spawns per-device scheduler actors
- spawns the memory actor
- creates `program_ptr` objects from source, PTX, cubin, or fatbin

Relevant files:

- `libcaf_cuda/caf/cuda/manager.hpp`
- `libcaf_cuda/src/manager.cpp`

### `platform`

`platform` is the device registry and system-wide CUDA runtime holder.

Responsibilities:

- discovers devices
- stores device objects
- routes scheduling requests to devices
- acts like the central runtime object behind `manager`

### `device`

`device` is the actual CUDA execution endpoint.

Responsibilities:

- creates GPU argument buffers as `mem_ptr<T>`
- owns CUDA streams and context access
- copies data host-to-device and device-to-host
- launches kernels with `cuLaunchKernel`

Relevant file:

- `libcaf_cuda/caf/cuda/device.hpp`

### `token_factory`

`token_factory` is a typed allocation layer for the control subsystem.

It creates the token objects that scheduler and memory actors exchange:

- launch requests
- launch responses
- memory requests
- memory responses
- transfer responses
- behavior-selection tokens

Relevant files:

- `libcaf_cuda/caf/cuda/control-layer/token_factory.hpp`
- `libcaf_cuda/src/control-layer/token_factory.cpp`

## Two practical launch paths

### Direct path

This is the simpler path used by high-level entry points such as `actor_facade` and parts of `command_runner`.

In this path:

- a command object is created
- arguments are converted into `mem_ptr<T>` objects
- `device::launch_kernel_mem_ref` or related execution helpers run the kernel
- `cuLaunchKernel` is called directly from the device layer

This path is more like a normal wrapper over CUDA driver calls.

### Control-layer path

This is the actor-driven scheduling path.

In this path:

- work is wrapped in tokens
- tokens are sent to scheduler actors
- scheduling behavior decides whether and when to admit work
- the memory actor may gate progress based on device memory pressure

This path is more experimental and policy-heavy.

## Token-driven launch flow

### Step 1: program creation

The user-facing code usually starts at `manager`.

`manager::create_program*` reads source or binary input and returns a `program_ptr` that represents a CUDA kernel loaded for execution.

### Step 2: request formation

The control layer packages a launch request into a `launch_token`.

That token carries the important scheduling metadata:

- which program to run
- the `nd_range`
- estimated memory usage
- an ID string
- the actor that should receive the reply
- an optional dependency

This packaging happens in `make_launch_token`.

### Step 3: scheduler actor receives work

Each GPU has a `scheduler_actor`.

The scheduler actor receives `token_ptr` messages and forwards them into the currently active scheduling behavior:

- `single_usage`
- `green_light`
- `red_light`
- `core_usage`
- others in the behavior table

The key point is that `scheduler_actor` itself is only a dispatch shell. The real policy decisions live in the behavior objects.

The `manager::send_scheduler_actor_message` helpers decide which scheduler actor receives the token:

- an explicit device number sends directly to that scheduler actor
- a dependent token is routed by dependency modulo device count
- an independent token is routed randomly

Relevant file:

- `libcaf_cuda/src/control-layer/scheduler_actor.cpp`
- `libcaf_cuda/src/manager.cpp`

### Step 4: memory coordination

The `memory_actor` tracks available memory per device.

When it receives a memory request token:

- if enough memory is available, it grants immediately
- otherwise, it queues the request

When memory is released later, the actor revisits the queue and grants pending requests in FIFO order.

This means the control layer is trying to serialize admission to the GPU based on memory pressure, not only based on kernel order.

Relevant file:

- `libcaf_cuda/src/control-layer/memory_actor/memory_actor.cpp`

### Step 5: execution on device

Once work is admitted and the right device/stream are chosen, the direct CUDA layer takes over:

- arguments are converted into `mem_ptr<T>` handles
- device memory is allocated when needed
- scalars are wrapped as scalar `mem_ref<T>` instances
- the kernel is launched via `cuLaunchKernel`

This is the point where the actor control layer ends and normal CUDA driver execution begins.

## Memory objects and why they matter

`mem_ptr<T>` is an intrusive pointer to `mem_ref<T>`.

`mem_ref<T>` represents either:

- a device allocation
- or a scalar value treated like a kernel argument object

This type is central because it connects actor messaging to actual GPU memory ownership.

That is why intrusive pointer consistency matters so much here: if the first reference is wrong, GPU memory can leak or be released at the wrong time.

## Architectural observations

### What is clean

- The system separates policy from execution reasonably well.
- The token types make the control-layer messages explicit.
- `memory_actor` gives you one clear place to reason about admission control.

### What is hard to reason about

- There are two schedulers in play conceptually: the CUDA `scheduler` abstraction and the actor-based `scheduler_actor` system.
- The token/control path and the direct `device` execution path overlap, which makes ownership and flow harder to trace.
- Some intrusive-pointer ownership rules are custom rather than idiomatic CAF, which increases maintenance cost.

## Ownership review notes

### `platform` and `device`

These are the closest parts to normal CAF style.

- `platform` is a singleton-like shared runtime object created through `make_counted`.
- `device` objects are also created through `make_counted` and then retained inside `platform`.
- Neither type introduces a second explicit `ref_count_{0}` field.

From an ownership perspective, these are the stable reference points in the current architecture.

### `program`

`program` is where the architecture stops looking CAF-like.

Conceptually, `program` is a long-lived kernel resource object. It holds:

- the kernel name
- binary/PTX/fatbin payload
- loaded kernel handles per device

But its lifetime model is mixed:

- it is created with `make_counted`
- it is passed around as `program_ptr`
- it overrides only part of the intrusive ownership behavior
- it carries a second counter and disabled deletion logic

That usually means one of two things:

1. the original author hit a real lifetime bug and patched around it locally
2. the resource boundary between CUDA module lifetime and C++ object lifetime was never fully stabilized

Either way, `program` should be treated as the main architectural review item.

### `mem_ref<T>`

`mem_ref<T>` is the bridge from actor-level messaging to concrete GPU memory ownership.

This type is unusual because it is both:

- a logical kernel argument wrapper
- the owner of a GPU allocation or scalar-backed pseudo-allocation

That makes it a sensitive type: if the ownership contract is fuzzy here, the entire launch path becomes hard to trust.

### `token` family

The token family is less dangerous than `program`, but still non-idiomatic.

It is effectively building a local intrusive ownership system for control-layer messages even though CAF already gives you one.

That may be workable, but it increases review burden because the reader must keep switching mental models.

## Reconciliation plan without changing code yet

If the goal is to move this closer to the rest of CAF, I would not start by editing constructors or replacing hooks. I would start by making the architecture explicit.

### Step 1: classify every major pointer-owning type

For each major heap type, decide which bucket it belongs in:

- CAF-owned lifecycle object
- CUDA resource owner
- control-layer message object
- temporary execution helper

Suggested classification pass:

1. `platform`
2. `device`
3. `program`
4. `mem_ref<T>`
5. `token` hierarchy
6. `command` hierarchy

### Step 2: identify the natural owner for each resource

This is the main question to answer before any code change:

- Should a `program` be owned by `manager`, by `actor_facade`, by commands, or by shared application code?
- Should a `mem_ref<T>` be owned only by returned handles, or can scheduler/control objects extend its lifetime too?
- Should tokens be pure message envelopes, or are they also resource owners?

If a type is both a message and a resource owner, that is usually where the model gets muddy.

### Step 3: reduce mixed responsibilities on paper first

Without changing code yet, write down the target split:

- resource objects own CUDA resources
- commands orchestrate execution
- tokens describe scheduling intent
- actors coordinate policy

That split is already partially present in the code. The problem is that the boundaries are not uniformly enforced.

### Step 4: pick a CAF-like endpoint

The most CAF-like destination would be:

- shared runtime/resource objects use standard CAF intrusive ownership
- control-layer message objects also use standard CAF intrusive ownership unless there is a proven reason not to
- object lifetime is described by pointer graphs, not by hidden manual refcount state

### Step 5: only then decide where code changes belong

Once the intended ownership graph is explicit, the actual change list becomes much safer to derive.

Likely future implementation phases would be:

1. normalize `program`
2. normalize `token`
3. normalize `mem_ref<T>` if needed
4. simplify or clearly separate the direct and control-layer launch paths

### What I would treat as the next review targets

1. `program` lifetime and ownership semantics.
2. The boundary between `scheduler_actor` decisions and `device` execution.
3. Whether the control-layer actor protocol is still buying enough value relative to its complexity.
4. Whether token objects should remain custom intrusive objects or become ordinary CAF-managed message objects.