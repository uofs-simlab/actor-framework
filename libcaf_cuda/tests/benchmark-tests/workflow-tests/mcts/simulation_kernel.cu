#include <cuda_runtime.h>
#include <curand_kernel.h>

// Must match the C++ definition exactly
struct game_state {
    int board[9]; 
    int turn;     
};

__device__ int check_winner(const int* board) {
    // Rows
    for (int i = 0; i < 9; i += 3)
        if (board[i] != 0 && board[i] == board[i+1] && board[i] == board[i+2]) return board[i];
    // Cols
    for (int i = 0; i < 3; ++i)
        if (board[i] != 0 && board[i] == board[i+3] && board[i] == board[i+6]) return board[i];
    // Diagonals
    if (board[0] != 0 && board[0] == board[4] && board[8]) return board[0];
    if (board[2] != 0 && board[2] == board[4] && board[6]) return board[2];

    // Check for draw
    bool full = true;
    for (int i = 0; i < 9; ++i) if (board[i] == 0) full = false;
    if (full) return 3; // 3 represents Draw

    return 0; // Ongoing
}

__device__ float perform_rollout(game_state state, curandState* local_state) {
    int current_turn = state.turn;
    int winner = 0;

    // Play randomly until terminal state
    for (int move = 0; move < 9; ++move) {
        winner = check_winner(state.board);
        if (winner != 0) break;

        // Find available moves
        int available[9];
        int count = 0;
        for (int i = 0; i < 9; ++i) {
            if (state.board[i] == 0) available[count++] = i;
        }

        if (count == 0) break;

        // Pick a random move
        int pick = curand(local_state) % count;
        state.board[available[pick]] = current_turn;
        current_turn = (current_turn == 1) ? 2 : 1;
    }

    if (winner == 3) return 0.5f; // Draw
    if (winner == 1) return 1.0f; // Player 1 wins
    return 0.0f;                 // Player 2 wins (or loss for P1)
}

extern "C" {

__global__ void evaluate_state(const game_state* initial_state, float* score_out, int seed_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize random number generator
    curandState local_state;
    curand_init(1234ULL, idx + seed_offset, 0, &local_state);

    // In this simple example, one thread does one rollout.
    // In a high-perf MCTS, one thread block might cooperate 
    // to do hundreds of rollouts for the same state.
    score_out[idx] = perform_rollout(*initial_state, &local_state);
}

} // extern "C"
