#!/bin/bash

set -euo pipefail

DATA_DIR="../data"

# Source prefix -> destination prefix
# 60xx
for i in {1..5}; do
    SRC=$(printf "%s/llm_run60%02d" "$DATA_DIR" "$i")

    # llama3.1: 6051-6055
    DST=$(printf "%s/llm_run605%d" "$DATA_DIR" "$i")
    mkdir -p "$DST"
    cp -R "$SRC/llama3.1:latest_baseline" "$DST/"
    cp -R "$SRC/llama3.1:latest_examples_rag" "$DST/"
    cp -R "$SRC/llama3.1:latest_examples_solution" "$DST/"

    # qwen3.5: 6061-6065
    DST=$(printf "%s/llm_run606%d" "$DATA_DIR" "$i")
    mkdir -p "$DST"
    cp -R "$SRC/qwen3.5:35b_baseline" "$DST/"
    cp -R "$SRC/qwen3.5:35b_examples_rag" "$DST/"
    cp -R "$SRC/qwen3.5:35b_examples_solution" "$DST/"

    # llama4: 6071-6075
    DST=$(printf "%s/llm_run607%d" "$DATA_DIR" "$i")
    mkdir -p "$DST"
    cp -R "$SRC/llama4:16x17b_baseline" "$DST/"
    cp -R "$SRC/llama4:16x17b_examples_rag" "$DST/"
    cp -R "$SRC/llama4:16x17b_examples_solution" "$DST/"
done

# 65xx
for i in {1..5}; do
    SRC=$(printf "%s/llm_run65%02d" "$DATA_DIR" "$i")

    # llama3.1: 6551-6555
    DST=$(printf "%s/llm_run655%d" "$DATA_DIR" "$i")
    mkdir -p "$DST"
    cp -R "$SRC/llama3.1:latest_baseline" "$DST/"
    cp -R "$SRC/llama3.1:latest_examples_rag" "$DST/"
    cp -R "$SRC/llama3.1:latest_examples_solution" "$DST/"

    # qwen3.5: 6561-6565
    DST=$(printf "%s/llm_run656%d" "$DATA_DIR" "$i")
    mkdir -p "$DST"
    cp -R "$SRC/qwen3.5:35b_baseline" "$DST/"
    cp -R "$SRC/qwen3.5:35b_examples_rag" "$DST/"
    cp -R "$SRC/qwen3.5:35b_examples_solution" "$DST/"

    # llama4: 6571-6575
    DST=$(printf "%s/llm_run657%d" "$DATA_DIR" "$i")
    mkdir -p "$DST"
    cp -R "$SRC/llama4:16x17b_baseline" "$DST/"
    cp -R "$SRC/llama4:16x17b_examples_rag" "$DST/"
    cp -R "$SRC/llama4:16x17b_examples_solution" "$DST/"
done

echo "Done."
