#!/usr/bin/env bash
# build_rag_mock_solution.sh
#
# Populates rag/mock/solution/<dataset>/<exercise_name>.{txt,bpmn}
# from data/models/.
#
# Usage (from repo root):
#   bash build_rag_mock_solution.sh
#
# Optionally override paths:
#   DATA_MODELS_DIR=./data/models RAG_OUT_DIR=./rag/mock/solution bash build_rag_mock_solution.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_MODELS_DIR="${DATA_MODELS_DIR:-$SCRIPT_DIR/../data/models}"
RAG_OUT_DIR="${RAG_OUT_DIR:-$SCRIPT_DIR/../rag/mock/solution}"

echo "Data models : $DATA_MODELS_DIR"
echo "Output root : $RAG_OUT_DIR"

# ---------------------------------------------------------------------------
# emit_single: one .txt + one known .bpmn file (no globbing)
#   $1 = dataset key
#   $2 = exercise stem
#   $3 = path to .txt file
#   $4 = path to .bpmn file
# ---------------------------------------------------------------------------
emit_single() {
    local dataset="$1" exercise="$2" txt_file="$3" bpmn_file="$4"
    local out_dir="$RAG_OUT_DIR/$dataset"
    mkdir -p "$out_dir"

    cp "$txt_file"  "$out_dir/$exercise.txt"
    echo "  [TXT]  $out_dir/$exercise.txt"
    cp "$bpmn_file" "$out_dir/$exercise.bpmn"
    echo "  [BPMN] $out_dir/$exercise.bpmn"
}

# ---------------------------------------------------------------------------
# emit: one .txt + a solution dir that may contain multiple *.bpmn files
#   $1 = dataset key  (e.g. "camunda", "camunda_2")
#   $2 = exercise name (becomes the output filename stem)
#   $3 = path to the .txt file
#   $4 = path to the solution dir containing *.bpmn
# ---------------------------------------------------------------------------
emit() {
    local dataset="$1"
    local exercise="$2"
    local txt_file="$3"
    local solution_dir="$4"

    local out_dir="$RAG_OUT_DIR/$dataset"
    mkdir -p "$out_dir"

    # --- text ---
    cp "$txt_file" "$out_dir/$exercise.txt"
    echo "  [TXT]  $out_dir/$exercise.txt"

    # --- bpmn(s) ---
    local bpmn_files=("$solution_dir"/*.bpmn)

    if [[ ! -e "${bpmn_files[0]}" ]]; then
        echo "  [WARN] No .bpmn found in $solution_dir — skipping BPMN for '$exercise'"
        return
    fi

    if [[ "${#bpmn_files[@]}" -eq 1 ]]; then
        cp "${bpmn_files[0]}" "$out_dir/$exercise.bpmn"
        echo "  [BPMN] $out_dir/$exercise.bpmn"
    else
        local i=1
        for bpmn in "${bpmn_files[@]}"; do
            cp "$bpmn" "$out_dir/${exercise}_${i}.bpmn"
            echo "  [BPMN] $out_dir/${exercise}_${i}.bpmn  (from $(basename "$bpmn"))"
            (( i++ ))
        done
    fi
}

# ---------------------------------------------------------------------------
# camunda / camunda_<N>
#   Full dataset:   data/models/camunda/.../English/<exercise>/01-Exercise/*.txt
#                                                              03-Solution/*.bpmn
#                               .../German/<exercise>/01-Aufgabenstellung/*.txt
#                                                     03-Musterlösung/*.bpmn
#   Versioned:      data/models/camunda_<N>/.../English/<exercise>/...  (English only)
# ---------------------------------------------------------------------------
process_camunda() {
    local version="${1:-}"          # empty = full dataset
    local dataset_key="camunda${version:+_$version}"

    echo ""
    echo "=== Processing dataset: $dataset_key ==="

    if [[ -n "$version" ]]; then
        # Versioned — English only
        local lang_dirs=(
            "$DATA_MODELS_DIR/camunda_${version}/bpmn-for-research-master/BPMN for Research/English|01-Exercise|03-Solution"
        )
    else
        # Full dataset — English + German
        local base="$DATA_MODELS_DIR/camunda/bpmn-for-research-master/BPMN for Research"
        local lang_dirs=(
            "$base/English|01-Exercise|03-Solution"
            "$base/German|01-Aufgabenstellung|03-Musterlösung"
        )
    fi

    for entry in "${lang_dirs[@]}"; do
        IFS='|' read -r lang_path text_subdir solution_subdir <<< "$entry"

        if [[ ! -d "$lang_path" ]]; then
            echo "  [SKIP] Directory not found: $lang_path"
            continue
        fi

        for ex_dir in "$lang_path"/*/; do
            [[ -d "$ex_dir" ]] || continue
            local exercise
            exercise="$(basename "$ex_dir")"
            [[ "$exercise" == .* ]] && continue

            # find the .txt (there's always exactly one)
            local txt_file
            txt_file="$(find "$ex_dir/$text_subdir" -maxdepth 1 -name "*.txt" | head -1)"
            if [[ -z "$txt_file" ]]; then
                echo "  [WARN] No .txt in $ex_dir/$text_subdir — skipping '$exercise'"
                continue
            fi

            emit "$dataset_key" "$exercise" "$txt_file" "$ex_dir/$solution_subdir"
        done
    done
}

# ---------------------------------------------------------------------------
# TODO stubs — paste the tree for each dataset and I'll fill these in
# ---------------------------------------------------------------------------
process_lre_new() {
    echo ""
    echo "=== Processing dataset: lre_new ==="

    local base="$DATA_MODELS_DIR/lre_new/NewDataset"
    local texts_dir="$base/Texts"
    local models_dir="$base/Models"

    if [[ ! -d "$texts_dir" ]]; then
        echo "  [SKIP] Directory not found: $texts_dir"
        return
    fi

    for txt_file in "$texts_dir"/*.txt; do
        [[ -f "$txt_file" ]] || continue
        local stem
        stem="$(basename "$txt_file" .txt)"
        local bpmn_file="$models_dir/$stem.bpmn"

        if [[ ! -f "$bpmn_file" ]]; then
            echo "  [WARN] No matching .bpmn for '$stem' — skipping"
            continue
        fi

        emit_single "lre_new" "$stem" "$txt_file" "$bpmn_file"
    done
}

process_lre_old() {
    # Dataset key is "lre_old" (matches get_datasets_config / prepare_lre_old),
    # but the folder on disk is lre_original/OriginalDataset — same layout as lre_new.
    echo ""
    echo "=== Processing dataset: lre_old ==="

    local base="$DATA_MODELS_DIR/lre_original/OriginalDataset"
    local texts_dir="$base/Texts"
    local models_dir="$base/Models"

    if [[ ! -d "$texts_dir" ]]; then
        echo "  [SKIP] Directory not found: $texts_dir"
        return
    fi

    for txt_file in "$texts_dir"/*.txt; do
        [[ -f "$txt_file" ]] || continue
        local stem
        stem="$(basename "$txt_file" .txt)"
        local bpmn_file="$models_dir/$stem.bpmn"

        if [[ ! -f "$bpmn_file" ]]; then
            echo "  [WARN] No matching .bpmn for '$stem' — skipping"
            continue
        fi

        emit_single "lre_old" "$stem" "$txt_file" "$bpmn_file"
    done
}

process_bpmn_and_text() {
    echo ""
    echo "=== Processing dataset: bpmn_and_text ==="

    local base="$DATA_MODELS_DIR/text_and_bpmn/bpmn"
    local out_dir="$RAG_OUT_DIR/bpmn_and_text"

    if [[ ! -d "$base" ]]; then
        echo "  [SKIP] Directory not found: $base"
        return
    fi

    mkdir -p "$out_dir"

    for txt_file in "$base"/*.txt; do
        [[ -f "$txt_file" ]] || continue

        local exercise
        exercise="$(basename "$txt_file" .txt)"

        local model_dir="$base/$exercise"

        if [[ ! -d "$model_dir" ]]; then
            echo "  [WARN] No model directory for '$exercise'"
            continue
        fi

        # Copy exercise description
        cp "$txt_file" "$out_dir/$exercise.txt"
        echo "  [TXT]  $out_dir/$exercise.txt"

        # Find BPMN with highest quality score
        local best_score=-1
        local best_bpmn=""

        for quality_file in "$model_dir"/*.quality.txt; do
            [[ -f "$quality_file" ]] || continue

            local score
            score="$(tr -d '[:space:]' < "$quality_file")"

            # Skip malformed scores
            [[ "$score" =~ ^[0-9]+$ ]] || continue

            if (( score > best_score )); then
                best_score="$score"

                local stem
                stem="$(basename "$quality_file" .quality.txt)"

                local candidate="$model_dir/${stem}.bpmn2.xml"
                if [[ -f "$candidate" ]]; then
                    best_bpmn="$candidate"
                fi
            fi
        done

        if [[ -z "$best_bpmn" ]]; then
            echo "  [WARN] No valid BPMN found for '$exercise'"
            continue
        fi

        cp "$best_bpmn" "$out_dir/$exercise.bpmn"
        echo "  [BPMN] $out_dir/$exercise.bpmn (quality=$best_score)"
    done
}

# ---------------------------------------------------------------------------
# Main — run all (or only the ones listed in $DATASETS)
# ---------------------------------------------------------------------------
ALL_DATASETS="camunda lre_new lre_old bpmn_and_text"
DATASETS="${DATASETS:-$ALL_DATASETS}"

for dataset in $DATASETS; do
    case "$dataset" in
        camunda)       process_camunda ;;
        camunda_[0-9]) process_camunda "${dataset#camunda_}" ;;
        lre_new)       process_lre_new ;;
        lre_old)       process_lre_old ;;
        bpmn_and_text) process_bpmn_and_text ;;
        *) echo "[ERROR] Unknown dataset '$dataset'" ;;
    esac
done

echo ""
echo "Done."