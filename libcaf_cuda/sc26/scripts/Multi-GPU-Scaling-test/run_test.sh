#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_DIR=/student/nqr159/data/scheduler-test/hetergenous-workloads/gpu_scaling_tests/scheduler
BIN="$SCRIPT_DIR/scheduler_test"

mkdir -p "$BASE_DIR"

declare -A GPU_CONFIGS
GPU_CONFIGS[1]="0"
GPU_CONFIGS[2]="0,1"
GPU_CONFIGS[4]="0,1,2,3"
GPU_CONFIGS[7]="0,1,2,3,4,5,6"
for gpus in 1 2 4 7; do
    echo "=============================="
    echo "Testing with $gpus GPU(s)"
    echo "=============================="

    OUT_DIR="$BASE_DIR/gpus_${gpus}"
    mkdir -p "$OUT_DIR"

    CUDA_DEVICES=${GPU_CONFIGS[$gpus]}

    for i in {1..10}; do
        echo "Running iteration $i with $gpus GPU(s)..."

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
        "$BIN" > "$OUT_DIR/output_${i}.txt"
    done
done
