# Intrusive Pointer Warning Report

## What the warning means

CAF deprecated this constructor:

```cpp
caf::intrusive_ptr<T>(raw_ptr, bool increase_ref_count = true)
```

The warning appears because code in the CUDA control layer still creates intrusive pointers directly from `new` expressions, for example:

```cpp
return token_ptr(new launch_token(...));
```

CAF now requires ownership to be explicit at construction time:

- `caf::add_ref` means: store the pointer and increment the intrusive reference count.
- `caf::adopt_ref` means: store the pointer without incrementing the intrusive reference count.

## Why these warnings occur here

The affected types in this codebase do **not** rely on CAF's default `ref_counted` counter. Instead, they define their own `intrusive_ptr_add_ref` and `intrusive_ptr_release` functions and keep a custom `ref_count_` initialized to `0`.

Relevant examples:

- `caf::cuda::token` in `libcaf_cuda/caf/cuda/control-layer/token.hpp`
- `caf::cuda::mem_ref<T>` in `libcaf_cuda/caf/cuda/mem_ref.hpp`

That means a freshly allocated object created with `new` starts with an effective intrusive reference count of `0` for the custom ownership path used by these classes.

Because of that, the first owning `intrusive_ptr` must use `caf::add_ref`, not `caf::adopt_ref`.

## What I fixed

I replaced the deprecated raw-pointer constructor usage with explicit `caf::add_ref` at the warning sites in:

- `libcaf_cuda/src/control-layer/token_factory.cpp`
- `libcaf_cuda/caf/cuda/device.hpp`
- `libcaf_cuda/src/scheduler.cpp`
- `libcaf_cuda/src/test.cpp`

The change preserves the old behavior while removing the deprecation warning.

Before:

```cpp
return token_ptr(new launch_token(...));
```

After:

```cpp
return token_ptr(new launch_token(...), caf::add_ref);
```

## Why `add_ref` is the correct fix

With the custom counters currently used in `token`, `mem_ref<T>`, and `program`, using `caf::adopt_ref` on a newly allocated object would incorrectly assume the object already owns one live reference.

In this code, that assumption is false because the custom intrusive count starts at `0`.

So:

- `caf::add_ref` changes the count from `0` to `1`, which matches the intended ownership of the first smart pointer.
- `caf::adopt_ref` would leave the count at `0`, which can lead to incorrect lifetime management.

## Token factory summary

`token_factory.cpp` is a small creation layer for the CUDA control subsystem. It constructs the token objects that the scheduler and memory-management actors pass around.

Main responsibilities:

- `make_launch_token`: creates a request token describing a kernel launch.
- `make_launch_response_token`: creates the scheduler's response token for an accepted launch.
- `make_behavior_token`: creates a token used for scheduler/control behavior selection.
- `make_memory_token`: creates a request token for memory transfer bookkeeping.
- `make_memory_response_token`: creates a response token for a completed memory transfer.
- `make_transfer_token`: creates a transfer-specific response token.
- `make_mem_ptr`: creates a test helper `mem_ptr<int>`.

Conceptually, the file centralizes token allocation so the rest of the control layer can work with typed handles instead of raw objects.

## Audit findings beyond the original warnings

### `mem_ref<T>` ownership in `device.hpp`

`device.hpp` was constructing `mem_ref<T>` objects with `caf::adopt_ref`.

That was inconsistent with the actual `mem_ref<T>` implementation, because `mem_ref<T>` uses a custom intrusive count that starts at `0`.

I changed these sites to `caf::add_ref` so the first owning pointer sets the count to `1`.

### `program` remains an architectural inconsistency

`program` in `libcaf_cuda/caf/cuda/program.hpp` still has mixed ownership behavior:

- it is constructed through `caf::make_counted`, which uses `caf::adopt_ref`
- it defines a custom `intrusive_ptr_release(const program*)`
- it also stores its own `ref_count_{0}`

That combination is not CAF-like and is a real correctness risk. I did **not** change it in this pass because it needs a wider review of `program` lifetime, especially since the current custom release path has deletion intentionally disabled.

If you want strict consistency across the project, `program` should be refactored next to use one ownership model only:

- either standard CAF `ref_counted`/`make_counted`
- or a fully custom intrusive refcount pair with matching construction semantics

Right now it mixes both.

## Ownership inconsistency inventory

This is the current ownership picture after warning cleanup.

### CAF-like and internally consistent

- `platform` behaves like normal CAF intrusive ownership: it uses `caf::ref_counted`, does not define a second manual counter, and uses `make_counted`.
- `device` also looks CAF-like: it is created with `make_counted` and does not redefine intrusive pointer hooks.
- `command`, `base_command`, and `memory_command` are also aligned with normal CAF construction patterns.

These types are not the current source of ownership ambiguity.

### Locally consistent but non-idiomatic

- `token` and token-derived types use a custom intrusive counter `ref_count_{0}` and custom `intrusive_ptr_add_ref` / `intrusive_ptr_release` hooks.
- `mem_ref<T>` uses the same pattern.

These can be made to work, but they are not CAF-like because they inherit from `caf::ref_counted` while bypassing its built-in counter.

This creates two overlapping lifetime systems in one type:

- CAF's `ref_counted` base counter
- the custom `ref_count_` field

Even if only one is active in practice, the type definition makes the ownership model harder to reason about.

### High-risk inconsistency

- `program` is the clearest mismatch in the project.

Why it stands out:

- it is created via `caf::make_counted`, which assumes CAF-style `adopt_ref` semantics
- it defines a custom `intrusive_ptr_release`
- it stores a second custom `ref_count_{0}`
- deletion in the custom release path is disabled

So `program` is neither fully CAF-style nor fully custom. It is the main place where the architecture should be reviewed before any further ownership cleanup.

## Concrete review plan

This is the order I would use if the goal is to reconcile the code with the rest of CAF while minimizing risk.

### Phase 1: write down the intended ownership contract per type

For each of these types, document three things before changing code:

- who creates it
- who owns the first strong reference
- who is responsible for the final release

Priority types:

1. `program`
2. `mem_ref<T>`
3. `token` and token subclasses
4. `platform`
5. `device`

For `platform` and `device`, the answer is mostly already clear. The purpose is to make the contrast explicit.

### Phase 2: verify the actual retention graph for `program`

Review these relationships carefully:

- `manager` creates `program_ptr`
- `actor_facade` stores `program_ptr`
- `command` and `base_command` store `program_ptr`
- `program::load_kernels` captures per-device kernel handles but does not currently release modules in a visible ownership abstraction

Questions to answer:

- Is `program` intended to live for the lifetime of the actor, for the lifetime of a command, or effectively forever?
- Is the disabled deletion in `program` hiding a real use-after-free bug elsewhere?
- Are CUDA module/function handles outliving the context or device they came from?

This phase should be review-only first. Do not change `program` until the lifetime graph is written down.

### Phase 3: choose one ownership model per type family

To follow CAF, each type family should use exactly one model.

Preferred target:

- `platform`, `device`, `program`, `command`, `memory_command`, `token`, and `mem_ref<T>` should each rely on CAF intrusive ownership without a second manual counter unless there is a compelling reason not to.

The key rule is simple:

- do not inherit from `caf::ref_counted` and also maintain a second independent refcount in the same object unless you have a very strong, documented reason.

### Phase 4: normalize construction semantics

Once each type has a single model, construction should follow directly from that choice.

- CAF-style objects created by `make_counted` should not need custom first-reference workarounds.
- Custom intrusive objects should not be mixed with `make_counted` unless their semantics are explicitly compatible.

This is where the code changes would eventually become mechanical.

### Phase 5: add lifetime-focused validation

Before changing behavior, add review checkpoints or tests for:

- repeated program creation and destruction
- actor teardown after launches
- stream release during actor destruction
- mem_ref scalar and buffer lifetime
- multi-device scheduling paths that retain tokens and responses longer than one call stack

## CAF-aligned reconciliation target

If the goal is to look like the rest of CAF, the most coherent target architecture is:

- `platform`, `device`, `program`, `command`, `memory_command`, `token`, and `mem_ref<T>` all use one CAF-style intrusive ownership model.
- `make_counted` creates CAF-owned objects.
- raw `new` plus explicit ownership tags is used only where that is truly necessary.
- no type carries both `caf::ref_counted` state and a second manual `ref_count_` field.

That target would make this codebase much easier to review because the ownership question would stop changing from class to class.

## What to investigate next, concretely

If you continue the review without changing code yet, I would inspect these files in this order:

1. `libcaf_cuda/caf/cuda/program.hpp` and `libcaf_cuda/src/program.cpp`
2. `libcaf_cuda/caf/cuda/command.hpp`
3. `libcaf_cuda/caf/cuda/actor_facade.hpp`
4. `libcaf_cuda/caf/cuda/mem_ref.hpp`
5. `libcaf_cuda/caf/cuda/control-layer/token.hpp`
6. `libcaf_cuda/src/manager.cpp`
7. `libcaf_cuda/caf/cuda/device.hpp`

The specific thing to look for in each file is not just “where is the pointer stored,” but “which object makes it safe for the pointee to die.”