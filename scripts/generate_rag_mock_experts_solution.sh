#!/usr/bin/env bash
# build_rag_mock_experts_solution.sh
#
# Populates rag/mock/experts_solution/experts/<exercise_name>.{txt,bpmn}
# from data_human_comparison/Expert_BPMN/.
#
# Source layout:
#   data_human_comparison/Expert_BPMN/textual_descriptions/<exercise>.txt
#   data_human_comparison/Expert_BPMN/ground_truth_bpmn/<exercise>.bpmn
#     (special case: "schufa" has two ground-truth variants,
#      schufa1.bpmn and schufa2.bpmn -- only schufa1.bpmn is used,
#      renamed to schufa.bpmn; schufa2.bpmn is dropped)
#
# Output layout:
#   rag/mock/experts_solution/experts/<exercise>.txt
#   rag/mock/experts_solution/experts/<exercise>.bpmn
#
# Usage (from repo root):
#   bash build_rag_mock_experts_solution.sh
#
# Optionally override paths:
#   EXPERT_BPMN_DIR=./data_human_comparison/Expert_BPMN \
#   RAG_OUT_DIR=./rag/mock/experts_solution \
#   bash build_rag_mock_experts_solution.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERT_BPMN_DIR="${EXPERT_BPMN_DIR:-$SCRIPT_DIR/../data_human_comparison/Expert_BPMN}"
RAG_OUT_DIR="${RAG_OUT_DIR:-$SCRIPT_DIR/../rag/mock/experts_solution}"

TEXTS_DIR="$EXPERT_BPMN_DIR/textual_descriptions"
GROUND_TRUTH_DIR="$EXPERT_BPMN_DIR/ground_truth_bpmn"
OUT_DIR="$RAG_OUT_DIR/experts"

echo "Expert BPMN dir : $EXPERT_BPMN_DIR"
echo "Output root     : $OUT_DIR"

if [[ ! -d "$TEXTS_DIR" ]]; then
    echo "[SKIP] Directory not found: $TEXTS_DIR"
    exit 0
fi

mkdir -p "$OUT_DIR"

for txt_file in "$TEXTS_DIR"/*.txt; do
    [[ -f "$txt_file" ]] || continue
    stem="$(basename "$txt_file" .txt)"

    cp "$txt_file" "$OUT_DIR/$stem.txt"
    echo "  [TXT]  $OUT_DIR/$stem.txt"

    # schufa special case: two ground-truth variants exist (schufa1/schufa2).
    # Only schufa1.bpmn is used, and it gets renamed to schufa.bpmn.
    # schufa2.bpmn is intentionally never copied.
    if [[ "$stem" == "schufa" ]]; then
        bpmn_file="$GROUND_TRUTH_DIR/schufa1.bpmn"
    else
        bpmn_file="$GROUND_TRUTH_DIR/$stem.bpmn"
    fi

    if [[ ! -f "$bpmn_file" ]]; then
        echo "  [WARN] No matching .bpmn for '$stem' ($bpmn_file) — skipping BPMN"
        continue
    fi

    cp "$bpmn_file" "$OUT_DIR/$stem.bpmn"
    echo "  [BPMN] $OUT_DIR/$stem.bpmn  (from $(basename "$bpmn_file"))"
done

echo ""
echo "Done."