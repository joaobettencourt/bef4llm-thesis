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

export LLMS=llama3.1:8b,llama3.3:70b,qwen3:14b
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
run_batch 111 112 113 114 115

# Mock RAG A
export RAG_ENABLED=true
export RAG_MODE=mock
export RAG_DIR=mock/set_a
run_batch 121 122 123 124 125

# Mock RAG B
export RAG_ENABLED=true
export RAG_MODE=mock
export RAG_DIR=mock/set_b
run_batch 131 132 133 134 135

# RAG A
export RAG_ENABLED=true
export RAG_MODE=examples
export RAG_DIR=set_a
run_batch 141 142 143 144 145

# RAG B
export RAG_ENABLED=true
export RAG_MODE=documents
export RAG_DIR=set_b
export RAG_TOP_K=3
export RAG_CHUNK_SIZE=500
export RAG_CHUNK_OVERLAP=100
run_batch 151 152 153 154 155