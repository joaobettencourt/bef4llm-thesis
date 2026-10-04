#!/bin/bash

set -euo pipefail

DATA_DIR="../data"

STAT_DATASETS_DIR="$DATA_DIR/statistical_datasets"
STAT_TESTS_DIR="$DATA_DIR/statistical_tests"

echo "========================================"
echo "Reorganizing statistical results"
echo "========================================"
echo


# ------------------------------------------------------------
# Helper: create statistical dataset directory
# ------------------------------------------------------------

copy_dataset() {
    local src_run="$1"
    local dst_run="$2"
    shift 2

    local src="$STAT_DATASETS_DIR/llm_metric_results_run${src_run}"
    local dst="$STAT_DATASETS_DIR/llm_metric_results_run${dst_run}"

    echo "Dataset:"
    echo "  $src"
    echo "  -> $dst"

    mkdir -p "$dst"

    for file in "$@"; do
        cp "$src/$file" "$dst/"
    done

    echo
}


# ------------------------------------------------------------
# Helper: create filtered statistical test group
# ------------------------------------------------------------

create_test_group() {
    local source_group="$1"
    local destination_group="$2"
    local col1="$3"
    local col2="$4"
    local col3="$5"

    local src="$STAT_TESTS_DIR/$source_group"
    local dst="$STAT_TESTS_DIR/$destination_group"

    echo "Statistical tests:"
    echo "  $source_group"
    echo "  -> $destination_group"

    mkdir -p "$dst"

    for metric in \
        "pragmatic quality.csv" \
        "syntactic quality.csv" \
        "semantic quality.csv"
    do
        python3 - "$src/$metric" "$dst/$metric" "$col1" "$col2" "$col3" <<'PY'
import csv
import sys

src = sys.argv[1]
dst = sys.argv[2]
wanted = sys.argv[3:]

with open(src, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f, delimiter=";")
    rows = list(reader)

header = rows[0]

# Always keep the first column: bpmn
indices = [0]

for column in wanted:
    if column not in header:
        raise RuntimeError(
            f"Column '{column}' not found in {src}"
        )
    indices.append(header.index(column))

filtered_rows = [
    [row[i] if i < len(row) else "" for i in indices]
    for row in rows
]

with open(dst, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(
        f,
        delimiter=";",
        lineterminator="\n"
    )
    writer.writerows(filtered_rows)
PY
    done

    echo
}


# ============================================================
# 1. STATISTICAL DATASETS
# ============================================================

echo "----------------------------------------"
echo "1. Statistical datasets"
echo "----------------------------------------"
echo


# ------------------------------------------------------------
# 60xx
# ------------------------------------------------------------

for i in {1..5}; do

    src_run=$(printf "600%d" "$i")

    # llama3.1: 6051-6055
    dst_run=$(printf "605%d" "$i")

    copy_dataset \
        "$src_run" \
        "$dst_run" \
        "llama3.1:latest_baseline.csv" \
        "llama3.1:latest_examples_rag.csv" \
        "llama3.1:latest_examples_solution.csv"

    # qwen3.5: 6061-6065
    dst_run=$(printf "606%d" "$i")

    copy_dataset \
        "$src_run" \
        "$dst_run" \
        "qwen3.5:35b_baseline.csv" \
        "qwen3.5:35b_examples_rag.csv" \
        "qwen3.5:35b_examples_solution.csv"

    # llama4: 6071-6075
    dst_run=$(printf "607%d" "$i")

    copy_dataset \
        "$src_run" \
        "$dst_run" \
        "llama4:16x17b_baseline.csv" \
        "llama4:16x17b_examples_rag.csv" \
        "llama4:16x17b_examples_solution.csv"

done


# ------------------------------------------------------------
# 65xx
# ------------------------------------------------------------

for i in {1..5}; do

    src_run=$(printf "650%d" "$i")

    # llama3.1: 6551-6555
    dst_run=$(printf "655%d" "$i")

    copy_dataset \
        "$src_run" \
        "$dst_run" \
        "llama3.1:latest_baseline.csv" \
        "llama3.1:latest_examples_rag.csv" \
        "llama3.1:latest_examples_solution.csv"

    # qwen3.5: 6561-6565
    dst_run=$(printf "656%d" "$i")

    copy_dataset \
        "$src_run" \
        "$dst_run" \
        "qwen3.5:35b_baseline.csv" \
        "qwen3.5:35b_examples_rag.csv" \
        "qwen3.5:35b_examples_solution.csv"

    # llama4: 6571-6575
    dst_run=$(printf "657%d" "$i")

    copy_dataset \
        "$src_run" \
        "$dst_run" \
        "llama4:16x17b_baseline.csv" \
        "llama4:16x17b_examples_rag.csv" \
        "llama4:16x17b_examples_solution.csv"

done


# ============================================================
# 2. STATISTICAL TESTS
# ============================================================

echo "----------------------------------------"
echo "2. Statistical tests"
echo "----------------------------------------"
echo

SOURCE_60="statistical_tests_group_6001_6002_6003_6004_6005"
SOURCE_65="statistical_tests_group_6501_6502_6503_6504_6505"


# ------------------------------------------------------------
# Llama 3.1
# ------------------------------------------------------------

create_test_group \
    "$SOURCE_60" \
    "statistical_tests_group_6051_6052_6053_6054_6055" \
    "llama3.1:latest_baseline" \
    "llama3.1:latest_examples_rag" \
    "llama3.1:latest_examples_solution"

create_test_group \
    "$SOURCE_65" \
    "statistical_tests_group_6551_6552_6553_6554_6555" \
    "llama3.1:latest_baseline" \
    "llama3.1:latest_examples_rag" \
    "llama3.1:latest_examples_solution"


# ------------------------------------------------------------
# Qwen 3.5
# ------------------------------------------------------------

create_test_group \
    "$SOURCE_60" \
    "statistical_tests_group_6061_6062_6063_6064_6065" \
    "qwen3.5:35b_baseline" \
    "qwen3.5:35b_examples_rag" \
    "qwen3.5:35b_examples_solution"

create_test_group \
    "$SOURCE_65" \
    "statistical_tests_group_6561_6562_6563_6564_6565" \
    "qwen3.5:35b_baseline" \
    "qwen3.5:35b_examples_rag" \
    "qwen3.5:35b_examples_solution"


# ------------------------------------------------------------
# Llama 4
# ------------------------------------------------------------

create_test_group \
    "$SOURCE_60" \
    "statistical_tests_group_6071_6072_6073_6074_6075" \
    "llama4:16x17b_baseline" \
    "llama4:16x17b_examples_rag" \
    "llama4:16x17b_examples_solution"

create_test_group \
    "$SOURCE_65" \
    "statistical_tests_group_6571_6572_6573_6574_6575" \
    "llama4:16x17b_baseline" \
    "llama4:16x17b_examples_rag" \
    "llama4:16x17b_examples_solution"


echo "========================================"
echo "Done."
echo "========================================"