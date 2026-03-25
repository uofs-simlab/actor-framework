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

## Follow-up note

There are existing `caf::adopt_ref` uses in `libcaf_cuda/caf/cuda/device.hpp` for `mem_ref<T>`. Those call sites should be audited separately, because with the current custom `mem_ref<T>` refcount implementation they appear inconsistent with the ownership model described above.