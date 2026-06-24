/*
 * A file full of caf cuda unit tests focused on the control layer of caf cuda
 */



#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/cuda/control-layer/all-control-layer.hpp>
#include <caf/actor_system.hpp>
#include "caf/cuda/control-layer/kernel_graph.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <cassert>

using namespace caf::cuda;

// Structure to hold test information
struct Test {
    std::string name;
    void (*function)(caf::actor_system&);
};


/*
 * kernel_graph_tests.cpp
 *
 * Unit tests for caf::cuda::kernel_graph
 *
 * These tests are lightweight and do not depend on launching kernels or
 * initializing the CUDA manager. They exercise the queue semantics:
 *   - peek() on an empty graph returns nullptr
 *   - add_operation() makes the graph non-empty
 *   - getOperation() pops the front element and restores empty state
 *
 * We deliberately use nullptr token_ptr instances so these tests do not
 * require constructing concrete token subclasses or having complex
 * control-layer dependencies present.
 */



// --- Tests ---

// 1) Validate empty graph behavior
void test_kernel_graph_empty([[maybe_unused]] caf::actor_system& sys) {
    kernel_graph g(/*device*/0, /*stream*/0, /*dependency*/0);

    // peek on empty graph -> should be nullptr
    auto p = g.peek();
    if (p != nullptr) {
        std::cerr << "[test_kernel_graph_empty] ERROR: peek() on empty graph returned non-null\n";
        throw std::runtime_error("peek() returned non-null on empty graph");
    }

    // getOperation on empty graph -> should be nullptr and graph remains empty
    auto r = g.getOperation();
    if (r != nullptr) {
        std::cerr << "[test_kernel_graph_empty] ERROR: getOperation() on empty graph returned non-null\n";
        throw std::runtime_error("getOperation() returned non-null on empty graph");
    }

    if (!g.empty()) {
        std::cerr << "[test_kernel_graph_empty] ERROR: graph reports non-empty after no ops\n";
        throw std::runtime_error("graph not empty after no ops");
    }

    std::cout << "[test_kernel_graph_empty] OK\n";
}

// 2) Validate push / pop semantics using nullptr token_ptr
void test_kernel_graph_push_pop([[maybe_unused]] caf::actor_system& sys) {
    kernel_graph g(/*device*/1, /*stream*/2, /*dependency*/3);

    // initially empty
    if (!g.empty()) {
        throw std::runtime_error("graph unexpectedly non-empty at start");
    }

    // create a token_ptr that is intentionally null to avoid heavy dependencies
    token_ptr t = nullptr;

    // add operation
    g.add_operation(t);

    // now graph should be non-empty
    if (g.empty()) {
        std::cerr << "[test_kernel_graph_push_pop] ERROR: graph empty after add_operation\n";
        throw std::runtime_error("graph empty after add_operation");
    }

    // peek should return the front element (nullptr in this test) but graph must not be empty
    auto pe = g.peek();
    // peek can be nullptr as we added a nullptr item — just ensure graph remains non-empty
    if (g.empty()) {
        std::cerr << "[test_kernel_graph_push_pop] ERROR: graph became empty after peek()\n";
        throw std::runtime_error("graph empty after peek");
    }

    // getOperation should remove and return the element (nullptr expected)
    auto popped = g.getOperation();
    //(void)popped; // we don't inspect the token itself in this test

    // after popping, graph should be empty again
    if (!g.empty()) {
        std::cerr << "[test_kernel_graph_push_pop] ERROR: graph not empty after getOperation\n";
        throw std::runtime_error("graph not empty after getOperation");
    }

    std::cout << "[test_kernel_graph_push_pop] OK\n";
}


void test_kernel_graph_can_move_initial([[maybe_unused]] caf::actor_system& sys) {
    kernel_graph g(/*device*/0, /*stream*/0);

    // Fresh graph should be allowed to move immediately
    if (!g.canMove()) {
        throw std::runtime_error("fresh kernel_graph cannot move");
    }

    std::cout << "[test_kernel_graph_can_move_initial] OK\n";
}

void test_kernel_graph_mark_moved_blocks([[maybe_unused]] caf::actor_system& sys) {
    kernel_graph g(/*device*/0, /*stream*/0);

    // First move allowed
    if (!g.canMove()) {
        throw std::runtime_error("kernel_graph cannot move initially");
    }

    // Mark as moved
    g.markMoved();

    // Immediately after marking, movement should be blocked
    if (g.canMove()) {
        throw std::runtime_error("kernel_graph canMove() returned true too soon after markMoved()");
    }

    std::cout << "[test_kernel_graph_mark_moved_blocks] OK\n";
}

void test_kernel_graph_can_move_after_delay([[maybe_unused]] caf::actor_system& sys) {
    kernel_graph g(/*device*/0, /*stream*/0);

    g.markMoved();

    // Sleep slightly longer than the default 2s threshold
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));

    if (!g.canMove()) {
        throw std::runtime_error("kernel_graph still blocked after delay");
    }

    std::cout << "[test_kernel_graph_can_move_after_delay] OK\n";
}



// 3) Test core_heuristic_function

void test_core_heuristic_function([[maybe_unused]] caf::actor_system& sys) {
    caf::cuda::manager::init(sys);
    auto& mgr = manager::get();
    auto dev = mgr.find_device(0);

    if (!dev) {
        // Skip test if no CUDA device
        return;
    }

    auto prog1 = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
    auto prog2 = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    nd_range r1(32, 32, 1, 32, 32, 1);
    nd_range r2(16, 16, 1, 16, 16, 1);

    core_heuristic_function h(dev);

    // Same program + same nd_range: cost should be consistent
    int cost1 = h.getCost(prog1, r1);
    int cost2 = h.getCost(prog1, r1);
    assert(cost1 == cost2);

    // Same program + different nd_range: cost should differ or be independent
    int cost3 = h.getCost(prog1, r2);
    assert(cost1 != ERROR_CODE);
    assert(cost3 != ERROR_CODE);

    // Different program + same nd_range: cost may differ
    int cost4 = h.getCost(prog2, r1);
    assert(cost4 != ERROR_CODE);

    // Cache stability: repeated calls give same result
    int cost5 = h.getCost(prog1, r1);
    int cost6 = h.getCost(prog1, r1);
    assert(cost5 == cost6);

    // Copy constructor preserves cache
    core_heuristic_function h_copy(dev, h);
    int cost7 = h_copy.getCost(prog1, r1);
    int cost8 = h_copy.getCost(prog1, r2);

    // Should match the original costs
    assert(cost7 == cost1);
    assert(cost8 == cost3);

    caf::cuda::manager::shutdown();
}







// Register tests
const std::vector<Test> tests = {
	{"test_kernel_graph_empty", test_kernel_graph_empty},
	{"test_kernel_graph_push_pop", test_kernel_graph_push_pop},
	{"test_kernel_graph_can_move_initial", test_kernel_graph_can_move_initial},
	{"test_kernel_graph_mark_moved_blocks", test_kernel_graph_mark_moved_blocks},
	{"test_kernel_graph_can_move_after_delay", test_kernel_graph_can_move_after_delay},
	{"test_core_heuristic_function", test_core_heuristic_function}

};

// Run a single test and return status code:
// 0 = PASS, 1 = SKIPPED, 2 = FAIL
int run_test(const Test& test, caf::actor_system& sys) {
    std::cout << "Running test: " << test.name << "... ";
    try {
        test.function(sys);
        std::cout << "PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << "\n";
        return 2;
    } catch (...) {
        std::cout << "FAILED: Unknown error\n";
        return 2;
    }
}

void caf_main(caf::actor_system& sys) {
    std::cout << "\nStarting kernel_graph unit tests...\n\n";
    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        int status = run_test(test, sys);
        if (status == 0) ++passed;
        else ++failed;
    }

    std::cout << "\nTest Summary:\n";
    std::cout << "Total tests listed: " << tests.size() << "\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";

    if (failed > 0) {
        throw std::runtime_error("One or more kernel_graph tests failed");
    }
}

CAF_MAIN()

