// #include <caf/all.hpp>
// #include <caf/cuda/all.hpp>
// #include <iostream>
// #include <deque>
// #include <vector>
// #include <random>
// #include <chrono>
// #include <filesystem>
// #include <memory>

// #include "caf/actorSOLVE/actorSOLVE.hpp"
// #include "sparse_utils.hpp"

// using namespace caf;
// using namespace caf::cuda;
// namespace fs = std::filesystem;

// // ============================================================
// // TYPE BLOCK (unchanged + extended)
// // ============================================================

// CAF_BEGIN_TYPE_ID_BLOCK(workload_test, caf::id_block::cuda::end)

//     CAF_ADD_ATOM(workload_test, get_work_atom)
//     CAF_ADD_ATOM(workload_test, request_work_atom)
//     CAF_ADD_ATOM(workload_test, worker_done_atom)
//     CAF_ADD_ATOM(workload_test, work_tick_atom)

//     CAF_ADD_TYPE_ID(workload_test, (SolverType))
//     CAF_ADD_TYPE_ID(workload_test, (MatrixTask))
//     CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
//     CAF_ADD_TYPE_ID(workload_test, (std::shared_ptr<MatrixData>))

// CAF_END_TYPE_ID_BLOCK(workload_test)

// CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixTask)
// CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<MatrixTask>)
// CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::shared_ptr<MatrixData>)

// // ============================================================
// // WORK TOKEN SYSTEM
// // ============================================================

// enum class Direction : uint8_t {
//     L2R = 0,
//     R2L = 1
// };

// inline Direction flip(Direction d) {
//     return d == Direction::L2R ? Direction::R2L : Direction::L2R;
// }

// struct WorkToken {
//     int device;
//     int stream;
//     size_t partition;
//     Direction dir;
// };

// template <class Inspector>
// bool inspect(Inspector& f, WorkToken& x) {
//     return f.object(x).fields(
//         f.field("device", x.device),
//         f.field("stream", x.stream),
//         f.field("partition", x.partition),
//         f.field("dir", x.dir)
//     );
// }

// // ============================================================
// // PARTITION STATE (supervisor owns)
// // ============================================================

// struct Partition {
//     size_t begin;
//     size_t end;

//     size_t active_tokens = 0;
//     bool completed = false;
// };

// // ============================================================
// // TASK ACTOR STATE
// // ============================================================

// struct task_state {
//     MatrixTask task;

//     actor supervisor;
//     actor left;
//     actor right;

//     bool seen_l2r = false;
//     bool seen_r2l = false;

//     bool solve_started = false;

//     std::optional<WorkToken> pending;

//     caf::actor cg_facade;

//     std::chrono::steady_clock::time_point start;
// };

// // ============================================================
// // TASK ACTOR
// // ============================================================

// behavior task_actor(stateful_actor<task_state>* self) {

//     return {

//         // ----------------------------------------------------
//         // INIT
//         // ----------------------------------------------------
//         [=](MatrixTask t,
//             actor supervisor,
//             actor left,
//             actor right) {

//             auto& st = self->state();

//             st.task = std::move(t);
//             st.supervisor = supervisor;
//             st.left = left;
//             st.right = right;

//             st.cg_facade =
//                 self->spawn<sparse_cg_facade<float>, linked>(0);
//         },

//         // ----------------------------------------------------
//         // TOKEN PROPAGATION
//         // ----------------------------------------------------
//         [=](const WorkToken& tok) {

//             auto& st = self->state();

//             bool& seen_this =
//                 tok.dir == Direction::L2R
//                     ? st.seen_l2r
//                     : st.seen_r2l;

//             bool& seen_other =
//                 tok.dir == Direction::L2R
//                     ? st.seen_r2l
//                     : st.seen_l2r;

//             // ------------------------------------------------
//             // COLLISION => PARTITION COMPLETED SIGNAL
//             // ------------------------------------------------
//             if (seen_other) {
//                 self->send(
//                     st.supervisor,
//                     worker_done_atom_v,
//                     tok.partition,
//                     tok.device,
//                     tok.stream
//                 );
//                 return;
//             }

//             // already visited this direction → just forward
//             if (seen_this) {
//                 actor next =
//                     tok.dir == Direction::L2R
//                         ? st.right
//                         : st.left;

//                 if (next)
//                     self->send(next, tok);

//                 return;
//             }

//             seen_this = true;

//             // ------------------------------------------------
//             // FIRST VISIT → LAUNCH SOLVER
//             // ------------------------------------------------
//             if (!st.solve_started) {
//                 st.solve_started = true;
//                 st.pending = tok;
//                 st.start = std::chrono::steady_clock::now();

//                 auto& d = *st.task.data;

//                 self->mail(
//                     create_in_arg(d.row_ptr),
//                     create_in_arg(d.col_indices),
//                     create_in_arg(d.values),
//                     create_in_arg(d.b),
//                     create_in_out_arg(d.x_guess),
//                     matrix_format::csr,
//                     (int)d.row_ptr.size() - 1,
//                     (int)d.values.size(),
//                     1e-5f,
//                     2000,
//                     tok.device,
//                     tok.stream
//                 ).send(st.cg_facade);

//                 return;
//             }

//             // ------------------------------------------------
//             // NORMAL FORWARD
//             // ------------------------------------------------
//             actor next =
//                 tok.dir == Direction::L2R
//                     ? st.right
//                     : st.left;

//             if (next)
//                 self->send(next, tok);
//         },

//         // ----------------------------------------------------
//         // SOLVER DONE
//         // ----------------------------------------------------
//         [=](uint32_t,
//             int,
//             std::vector<float>&,
//             solver_result_meta meta) {

//             auto& st = self->state();

//             auto tok = *st.pending;

//             actor next =
//                 tok.dir == Direction::L2R
//                     ? st.right
//                     : st.left;

//             if (next) {
//                 self->send(next, tok);
//             } else {
//                 self->send(
//                     st.supervisor,
//                     worker_done_atom_v,
//                     tok.partition,
//                     tok.device,
//                     tok.stream
//                 );
//             }
//         }
//     };
// }

// // ============================================================
// // SUPERVISOR STATE
// // ============================================================

// struct supervisor_state {
//     std::vector<MatrixTask> batch;
//     std::vector<actor> actors;
//     std::vector<Partition> partitions;

//     size_t next_partition = 0;
//     size_t completed = 0;

//     int num_streams = 0;
// };

// // ============================================================
// // SUPERVISOR
// // ============================================================

// behavior supervisor_actor(stateful_actor<supervisor_state>* self,
//                           std::vector<MatrixTask> batch,
//                           int num_gpus,
//                           int streams_per_gpu) {

//     auto& st = self->state();
//     st.batch = std::move(batch);

//     size_t num_parts = num_gpus * streams_per_gpu;

//     st.partitions.resize(num_parts);

//     for (size_t i = 0; i < num_parts; ++i) {
//         size_t begin = (i * st.batch.size()) / num_parts;
//         size_t end   = ((i + 1) * st.batch.size()) / num_parts;

//         st.partitions[i] = {begin, end, 0, false};
//     }

//     // spawn actors
//     int total = st.batch.size();

//     for (int i = 0; i < total; ++i)
//         st.actors.push_back(self->spawn(task_actor));

//     // link neighbors
//     for (size_t i = 0; i < st.actors.size(); ++i) {
//         actor left  = (i == 0) ? actor{} : st.actors[i - 1];
//         actor right = (i + 1 < st.actors.size()) ? st.actors[i + 1] : actor{};

//         self->send(st.actors[i],
//                    st.batch[i],
//                    actor_cast<actor>(self),
//                    left,
//                    right);
//     }

//     // inject first wave
//     auto& p = st.partitions[0];

//     p.active_tokens = 1;

//     self->send(
//         st.actors[p.begin],
//         WorkToken{
//             0,
//             0,
//             0,
//             Direction::L2R
//         }
//     );

//     return {

//         // ------------------------------------------------
//         // TOKEN COMPLETION EVENT
//         // ------------------------------------------------
//         [=](worker_done_atom,
//             size_t partition,
//             int device,
//             int stream) {

//             auto& st = self->state();

//             auto& p = st.partitions[partition];

//             if (p.completed)
//                 return;

//             if (p.active_tokens > 0)
//                 p.active_tokens--;

//             if (p.active_tokens == 0) {
//                 p.completed = true;
//                 st.completed++;

//                 std::cout << "[DONE] partition "
//                           << partition << "\n";
//             }

//             // find next unfinished
//             for (size_t i = 0; i < st.partitions.size(); ++i) {
//                 size_t idx = (partition + i + 1)
//                            % st.partitions.size();

//                 if (!st.partitions[idx].completed) {

//                     auto& np = st.partitions[idx];
//                     np.active_tokens++;

//                     Direction dir =
//                         (i % 2 == 0)
//                             ? Direction::L2R
//                             : Direction::R2L;

//                     size_t start =
//                         (dir == Direction::L2R)
//                             ? np.begin
//                             : np.end - 1;

//                     self->send(
//                         st.actors[start],
//                         WorkToken{
//                             device,
//                             stream,
//                             idx,
//                             dir
//                         });

//                     break;
//                 }
//             }
//         }
//     };
// }

// // ============================================================
// // MAIN
// // ============================================================

// void caf_main(actor_system& sys) {

//     manager::init(sys, manager_config(true, true));

//     int streams = 8;
//     int batches = 25;
//     int batch_size = 100;

//     auto tasks = scan_for_matrices(
//         "/scratch/nqr159/matrix-collection",
//         CGS_SOLVER
//     );

//     auto batch = generate_batch(tasks, std::mt19937{42}, batch_size);

//     manager& mgr = manager::get();
//     int gpus = mgr.get_num_devices();

//     sys.spawn(supervisor_actor,
//               batch,
//               gpus,
//               streams);

//     sys.await_all_actors_done();

//     manager::shutdown();
// }

// CAF_MAIN(id_block::cuda, workload_test)