# Actor Facade Analysis

## Summary

The current CUDA actor facade works correctly for the request/reply flow used by the examples, but it is **not fully non-blocking from CAF's scheduler perspective**.

What it does well:

- The facade can be spawned through the existing manager API.
- It accepts kernel arguments as a CAF message.
- It launches the GPU work and returns a `std::vector<output_buffer>` to the requester.
- The five example builds are now clean after the warning fixes.

What it does not do yet:

- It does **not** yield the CAF worker thread while waiting for GPU completion.
- It does **not** get rescheduled by a CUDA completion event.
- It performs a synchronous wait when copying results back to host memory.

That means the implementation is functionally correct, but not yet aligned with your target execution model of:

1. Spawn facade.
2. Send GPU work.
3. Let the actor stop consuming a CAF worker while the GPU runs.
4. Re-enter CAF only when the GPU finishes.

## Files Involved

- [libcaf_cuda/caf/cuda/actor_facade_event.hpp](libcaf_cuda/caf/cuda/actor_facade_event.hpp)
- [libcaf_cuda/caf/cuda/actor_facade.hpp](libcaf_cuda/caf/cuda/actor_facade.hpp)
- [libcaf_cuda/caf/cuda/manager.hpp](libcaf_cuda/caf/cuda/manager.hpp)
- [libcaf_cuda/caf/detail/spawn_helper.hpp](libcaf_cuda/caf/detail/spawn_helper.hpp)
- [libcaf_cuda/caf/cuda/command.hpp](libcaf_cuda/caf/cuda/command.hpp)
- [libcaf_cuda/caf/cuda/device.hpp](libcaf_cuda/caf/cuda/device.hpp)
- [libcaf_cuda/caf/cuda/mem_ref.hpp](libcaf_cuda/caf/cuda/mem_ref.hpp)
- [libcaf_core/caf/scheduled_actor.cpp](libcaf_core/caf/scheduled_actor.cpp)
- [libcaf_core/caf/detail/default_invoke_result_visitor.hpp](libcaf_core/caf/detail/default_invoke_result_visitor.hpp)
- [libcaf_core/caf/local_actor.hpp](libcaf_core/caf/local_actor.hpp)
- [libcaf_core/caf/response_promise.hpp](libcaf_core/caf/response_promise.hpp)

## How Spawn Works

The user-facing spawn path is:

1. `manager::spawn` or `manager::spawnFromCUBIN` constructs a `program_ptr` and `nd_range`.
2. It calls `caf::detail::cuda_spawn_helper`.
3. `cuda_spawn_helper` calls `actor_facade::create(...)`.
4. `actor_facade::create(...)` spawns `actor_facade`, which inherits from `actor_facade_event`.

The important part is that the public `actor_facade` type is now just a compatibility wrapper. The real behavior lives in `actor_facade_event`.

## How the Facade Receives Work

`actor_facade_event` is a normal `caf::event_based_actor`. Its `make_behavior()` returns a single handler that accepts a `caf::message`.

Inside that handler:

1. `handle_message(msg)` checks which message shape arrived.
2. It accepts four formats:
   - `Ts...`
   - `raw_t<Ts>...`
   - `caf::actor, Ts...`
   - `caf::actor, raw_t<Ts>...`
3. It unpacks the values into the expected CUDA wrapper types.
4. It calls `launch_kernel(...)`.

`launch_kernel(...)` constructs a `command<caf::actor, ...>` and calls `enqueue()`.

## How the Kernel Launch Works

The launch path is:

1. `actor_facade_event::launch_kernel(...)`
2. `command::enqueue()`
3. `base_command::base_enqueue()`
4. `device::launch_kernel_mem_ref(...)`
5. `device::collect_output_buffers(...)`
6. `mem_ref::copy_to_host()`

Two details matter here:

- `device::launch_kernel_mem_ref(...)` launches the CUDA kernel asynchronously onto a stream.
- `mem_ref::copy_to_host()` then performs `cuMemcpyDtoHAsync(...)` followed by `cuStreamSynchronize(...)`.

So the kernel launch starts asynchronously, but the facade immediately waits for completion during result collection.

## How the Reply Gets Back to the Requesting Actor

This is the part that is easy to miss: the facade does **not** explicitly call `send(sender, result)`.

Instead, CAF handles the reply automatically because the behavior handler returns a value.

The chain is:

1. The requester uses `.request(gpuActor, timeout)`.
2. CAF delivers that request as a mailbox element carrying sender information and a request/response message ID.
3. `scheduled_actor::consume(...)` invokes the behavior for that mailbox element.
4. The handler in `actor_facade_event::make_behavior()` returns `caf::result<std::vector<output_buffer>>`.
5. CAF's `default_invoke_result_visitor` turns that return value into `self_->respond(...)`.
6. `local_actor::respond(...)` uses `response_promise::respond_to(this, current_mailbox_element(), x)`.
7. CAF sends the response to the original requester using the metadata stored in the current mailbox element.

So the facade "knows" where to send the result back because CAF tracks the active request while the handler is running.

The sender and response ID are already attached to the mailbox element for the current message.

## Important Consequence

Automatic replies only work in this direct way because the result is produced while the actor is still handling the original request.

If you later change this facade to launch work truly asynchronously and return later, you will need to capture a `response_promise` explicitly and fulfill it when the GPU work completes.

## Does the Facade Hold a CAF Worker Thread While Waiting?

Yes.

The current implementation blocks a CAF worker while handling each GPU request.

Why:

1. The event-based actor is scheduled on a CAF worker.
2. The message handler runs on that worker.
3. The handler calls `command::enqueue()`.
4. `command::enqueue()` collects output buffers synchronously.
5. `mem_ref::copy_to_host()` calls `cuStreamSynchronize(...)`.
6. The CAF worker remains occupied until the GPU stream finishes and the copy back to host completes.

This means the actor is event-based in type, but the actual request handler is still synchronous end-to-end.

## Does It Get Rescheduled When the GPU Finishes?

No, not in the way you want.

The actor is not suspended and later awakened by a GPU completion signal.

What really happens is:

1. The actor starts handling the request.
2. It launches the GPU kernel.
3. It blocks waiting for the stream to finish.
4. It copies the result back.
5. It returns the result.
6. The handler ends.

So there is no separate completion message, no callback into CAF, and no fresh scheduling point when the GPU completes.

## Is It Prone to Starvation?

### Within the facade itself

There is head-of-line blocking.

Because a single actor processes one message at a time, any long-running GPU request prevents the same facade instance from starting the next request until the first request finishes.

This is expected actor behavior, but it means one slow kernel can delay all later requests queued to that same facade.

### In the CAF worker pool

There is a real risk of worker starvation under load.

If many facades or many GPU-heavy requests are active at once, each one can occupy a CAF scheduler thread while waiting in `cuStreamSynchronize(...)`.

That can reduce the number of workers available to unrelated actors and cause:

- increased latency
- unfairness under load
- reduced throughput
- possible starvation symptoms for other actors if the worker pool is small relative to the number of blocked GPU requests

### On the GPU side

The facade uses per-actor streams via `actor_id_`, which helps separate GPU work logically, but stream separation alone does not solve CAF worker starvation because the host thread still waits synchronously.

## Does the Current Design Match the Desired Model?

No.

Your desired model is:

1. facade receives request
2. facade launches GPU work
3. facade releases the CAF worker
4. GPU completion later causes a message back into CAF
5. facade sends reply to requester

The current model is:

1. facade receives request
2. facade launches GPU work
3. facade waits synchronously for completion
4. facade sends reply before the handler returns

So the current design is correct for functional behavior, but not for asynchronous scheduler behavior.

## What Would Be Needed for a Truly Non-Blocking Facade

The facade would need to switch from "return result directly from the request handler" to "capture a response promise and fulfill it later".

A better design would be:

1. On request arrival, create a `response_promise`.
2. Launch GPU work using the asynchronous path, such as the existing `base_command` / `run_async` style flow.
3. Do not call `collect_output_buffers()` inside the request handler.
4. Register a completion mechanism:
   - CUDA event + polling actor, or
   - detached thread waiting on the event, or
   - CUDA host callback if the driver/runtime combination supports it safely in this project
5. When the GPU completes, send a normal CAF message back to the facade containing either:
   - the completed `mem_ptr` tuple, or
   - already copied host-side outputs
6. On that follow-up message, fulfill the stored `response_promise`.

That version would allow the original actor handler to return immediately and free the CAF worker.

## Suggested Direction

If your next goal is scheduler-friendly GPU integration, the facade should evolve into a two-stage actor:

1. **submission stage**
   - receives request
   - creates response promise
   - launches GPU work asynchronously
   - stores pending request state

2. **completion stage**
   - receives completion notification
   - copies results back if needed
   - delivers the stored response promise

That would match the execution model you described much more closely.

## Warning Fixes Completed

The warnings you were seeing in the five example builds were fixed by:

- adding the full `program.hpp` include to `command.hpp` so `program_->get_kernel(...)` no longer uses an incomplete type
- replacing deprecated `caf::intrusive_ptr(ptr, bool)` construction in `device.hpp` with `caf::adopt_ref`

After reinstalling CAF and rebuilding `Actor_Tests` from clean, all five example builds completed without the previously reported warnings.