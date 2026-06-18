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
        python src/compare_llms.py generate_bpmn "$run"
    done
}

# Baseline
export RAG_ENABLED=false
#run_batch 111 112 113

# Mock RAG A
export RAG_ENABLED=true
export RAG_MODE=mock_examples
export RAG_DIR=mock/solution
run_batch 121 122 123
