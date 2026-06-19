#!/bin/bash

set -e

DATASET_NAME="2"

if [ ! -d "../data/models/camunda_${DATASET_NAME}" ]; then
    echo "camunda_${DATASET_NAME} dataset not found. Generating it..."
    ./generate_camunda_subset.sh "$DATASET_NAME" \
        01-Dispatch-of-goods \
        02-Recourse
fi

cd ..

export LLMS=llama3.1:latest,qwen2.5:latest
export DATASET_MODE=general
export DATASETS=camunda_2

run_batch() {
    for run in "$@"; do
        echo "Generating BPMNs for run $run"
        log_dir="data/llm_run${run}"
        mkdir -p "$log_dir"
        python src/compare_llms.py generate_bpmn "$run" 2>&1 | tee "$log_dir/run_${run}.log"
    done
}

join_batch() {
    local suffix=$1
    local run1=$2
    local run2=$3
    local run3=$4

    cd scripts
    ./join_runs.sh 101 "$run1" "$suffix"
    ./join_runs.sh 102 "$run2" "$suffix"
    ./join_runs.sh 103 "$run3" "$suffix"
    cd ..
}

# Baseline
export RAG_ENABLED=false
run_batch 111 112 113
join_batch baseline 111 112 113

# Mock RAG example solution
export RAG_ENABLED=true
export RAG_MODE=mock_examples
export RAG_DIR=mock/solution
run_batch 121 122 123
join_batch mock_examples_solution 121 122 123

# Mock RAG example stakeholders
export RAG_ENABLED=true
export RAG_MODE=mock_examples
export RAG_DIR=mock/stakeholders
run_batch 131 132 133
join_batch mock_examples_stakeholders 131 132 133
