#!/bin/bash
set -e

# Usage: ./join_runs.sh <dest_run> <src_run> <suffix>
# Example: ./join_runs.sh 101 121 mock_examples_solution

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dest_run> <src_run> <suffix>"
    echo "Example: $0 101 121 mock_examples_solution"
    exit 1
fi

DEST_RUN=$1
SRC_RUN=$2
SUFFIX=$3

SRC_DIR="../data/llm_run${SRC_RUN}"
DEST_DIR="../data/llm_run${DEST_RUN}"

if [ ! -d "$SRC_DIR" ]; then
    echo "[ERROR] Source run directory not found: $SRC_DIR"
    exit 1
fi

mkdir -p "$DEST_DIR"
echo "[INFO] Destination: $DEST_DIR"

for llm_dir in "$SRC_DIR"/*/; do
    llm_name=$(basename "$llm_dir")
    dest_llm_dir="$DEST_DIR/${llm_name}_${SUFFIX}"

    if [ -d "$dest_llm_dir" ]; then
        echo "[WARNING] Destination already exists, skipping: $dest_llm_dir"
        continue
    fi

    echo "[INFO] Copying $llm_name -> ${llm_name}_${SUFFIX}"
    cp -r "$llm_dir" "$dest_llm_dir"
done

echo "[DONE] Run $SRC_RUN copied into $DEST_DIR with suffix '_${SUFFIX}'"