/*
 * A file full of caf cuda unit tests focused on the control layer of caf cuda
 */



#include <caf/all.hpp>
#include <caf/actor_system.hpp>
#include "caf/cuda/control-layer/kernel_graph.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

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

// Register tests
const std::vector<Test> tests = {
    {"test_kernel_graph_empty", test_kernel_graph_empty},
    {"test_kernel_graph_push_pop", test_kernel_graph_push_pop}
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

