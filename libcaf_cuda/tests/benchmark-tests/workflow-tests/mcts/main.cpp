#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace caf;
using namespace std::chrono_literals;

// Atoms for MCTS operations
using search_atom = atom_constant<uint32_t, 1>;
using simulate_atom = atom_constant<uint32_t, 2>;
using update_atom = atom_constant<uint32_t, 3>;
using expand_atom = atom_constant<uint32_t, 4>;

// Mock Game State
struct game_state {
    int board[9]; // e.g., Tic-Tac-Toe
    int turn;     // 1 or 2

    template <class Inspector>
    friend bool inspect(Inspector& f, game_state& x) {
        return f.object(x).fields(f.field("board", x.board), f.field("turn", x.turn));
    }
};

// Command runner for the GPU Simulation kernel
// The kernel takes the current game state and returns a win probability (float)
using simulation_command = caf::cuda::command_runner<
    in<game_state>, // Input game state
    out<float>,     // Output score
    in<int>         // Seed offset
>;

struct mcts_node_state {
    game_state state;
    caf::actor parent;
    std::vector<caf::actor> children;
    
    int visits = 0;
    float total_value = 0.0f;
    bool is_expanded = false;

    caf::cuda::program_ptr sim_program;

    float ucb1(int parent_visits) const {
        if (visits == 0) return 1e9f; // Priority for unvisited nodes
        return (total_value / visits) + 1.41f * std::sqrt(std::log(parent_visits) / visits);
    }
};

// The MCTS Node Actor
caf::behavior mcts_node_fun(caf::stateful_actor<mcts_node_state>* self, 
                            caf::actor parent, 
                            game_state state,
                            caf::cuda::program_ptr sim_prog) {
    self->state().parent = parent;
    self->state().state = state;
    self->state().sim_program = sim_prog;

    return {
        // Traversal / Selection Phase
        [=](search_atom) {
            if (!self->state().is_expanded) {
                // Leaf node reached: Trigger Simulation on GPU
                self->mail(simulate_atom_v).send(self);
            } else if (self->state().children.empty()) {
                // Terminal node or no moves possible
                self->mail(update_atom_v, 0.5f).send(self);
            } else {
                // Select best child using UCB1
                auto it = std::max_element(self->state().children.begin(), 
                                         self->state().children.end(),
                                         [=](const caf::actor& a, const caf::actor& b) {
                                             // Note: In a real app, you'd request stats from children 
                                             // or cache them in the parent state for speed.
                                             return true; 
                                         });
                self->mail(search_atom_v).send(*it);
            }
        },

        // Simulation Phase: Launching CUDA Kernel
        [=](simulate_atom) {
            simulation_command cmd;
            int device = 0;
            int actor_id = static_cast<int>(self->id());

            // Prepare GPU arguments
            auto in_state = caf::cuda::create_in_arg(self->state().state);
            auto out_score = caf::cuda::create_out_arg_with_size<float>(1);
            auto seed_arg = caf::cuda::create_in_arg(actor_id);

            // Configure Kernel Dims (1 block, 1 thread for a single simulation rollout)
            // In a real scenario, you'd run many rollouts in parallel on the GPU.
            caf::cuda::nd_range dims(1, 1, 1, 1, 1, 1);

            auto results = cmd.run(self->state().sim_program, dims, actor_id, in_state, out_score, seed_arg);
            float win_rate = caf::cuda::extract_vector<float>(results)[0];

            // After GPU returns result, trigger Expansion and Backpropagation
            self->mail(expand_atom_v).send(self);
            self->mail(update_atom_v, win_rate).send(self);
        },

        // Expansion Phase: Spawning child actors for new moves
        [=](expand_atom) {
            if (self->state().is_expanded) return;

            // Mock expansion: spawn 3 child actors representing possible moves
            for (int i = 0; i < 3; ++i) {
                game_state next_state = self->state().state;
                next_state.turn = (next_state.turn == 1) ? 2 : 1;
                
                auto child = self->spawn(mcts_node_fun, 
                                       caf::actor_cast<caf::actor>(self), 
                                       next_state, 
                                       self->state().sim_program);
                self->state().children.push_back(child);
            }
            self->state().is_expanded = true;
        },

        // Backpropagation Phase
        [=](update_atom, float result) {
            self->state().visits++;
            self->state().total_value += result;

            if (self->state().parent) {
                // Pass result up the tree
                self->mail(update_atom_v, result).send(self->state().parent);
            } else {
                std::cout << "[Root] Search iteration complete. Root visits: " 
                          << self->state().visits << std::endl;
            }
        }
    };
}

void run_mcts_demo(caf::actor_system& sys) {
    caf::cuda::manager::init(sys);
    auto& mgr = caf::cuda::manager::get();

    // Load the simulation kernel
    // This kernel would perform random rollouts from the given state
    auto program = mgr.create_program_from_cubin("simulation_kernel.cubin", "evaluate_state");

    game_state initial_state;
    initial_state.turn = 1;
    for(int& i : initial_state.board) i = 0;

    // Spawn the Root actor
    auto root = sys.spawn(mcts_node_fun, nullptr, initial_state, program);

    std::cout << "Starting MCTS iterations..." << std::endl;

    // Run 100 search iterations
    for (int i = 0; i < 100; ++i) {
        caf::anon_mail(search_atom_v).send(root);
    }

    // Wait for some results
    std::this_thread::sleep_for(2s);

    // Demonstration of pruning: kill the root to stop the whole tree
    caf::anon_mail(exit_msg{root, exit_reason::user_shutdown}).send(root);
    
    sys.await_all_actors_done();
    caf::cuda::manager::shutdown();
    std::cout << "MCTS Demo finished." << std::endl;
}

void caf_main(caf::actor_system& sys) {
    run_mcts_demo(sys);
}

CAF_MAIN(caf::cuda::id_block)
