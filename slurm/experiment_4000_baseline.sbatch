#!/bin/bash
#SBATCH --job-name=experiment_4000_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=gpu075522026
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/users5/cpca075522026/jbettencourt/bef4llm-thesis/slurm/logs/%x_%j.out
#SBATCH --error=/users5/cpca075522026/jbettencourt/bef4llm-thesis/slurm/logs/%x_%j.err

set -e

# =========================================================
# Project root (do not rely on --chdir)
# =========================================================
PROJECT_ROOT="/users5/cpca075522026/jbettencourt/bef4llm-thesis"
cd "$PROJECT_ROOT"

# =========================================================
# 0. Reorganize logs into a per-run folder
# =========================================================
RUN_DIR="$PROJECT_ROOT/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

mv "$PROJECT_ROOT/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$RUN_DIR/slurm.out" 2>/dev/null || true
mv "$PROJECT_ROOT/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$RUN_DIR/slurm.err" 2>/dev/null || true
cp "$0" "$RUN_DIR/run.sbatch" 2>/dev/null || true

echo "[$(date)] Working directory: $(pwd)"
echo "[$(date)] Logs for this run: $RUN_DIR"

# =========================================================
# 1. Start Ollama and wait until it is ready
# =========================================================
echo "[$(date)] Starting ollama serve..."
export OLLAMA_LOAD_TIMEOUT=30m
ollama serve > "$RUN_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!

cleanup() {
    echo "[$(date)] Stopping ollama (pid $OLLAMA_PID)..."
    kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[$(date)] Waiting for ollama to become ready..."
READY=false
for i in $(seq 1 60); do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "[$(date)] Ollama is up."
        READY=true
        break
    fi
    sleep 2
done

if [ "$READY" != "true" ]; then
    echo "Ollama did not become ready in time." >&2
    exit 1
fi

# =========================================================
# 2. Activate the Python environment
# =========================================================
source ~/fw-env/bin/activate

# =========================================================
# 3. Shared config
# =========================================================
export OLLAMA_HOST=http://localhost:11434
export LLMS=llama4:16x17b,qwen3.5:35b,eurollm-22b:latest
export DATASET_MODE=general
#export DATASETS=camunda,lre_new,lre_original,text_and_bpmn
export DATASETS=camunda,lre_new,lre_old,bpmn_and_text

run_batch() {
    for run in "$@"; do
        echo "[$(date)] Generating BPMNs for run $run"
        log_dir="data/llm_run${run}"
        mkdir -p "$log_dir"
        python src/compare_llms.py generate_bpmn "$run" 2>&1 | tee "$RUN_DIR/run_${run}.log"
    done
}

# =========================================================
# 4. Baseline (no RAG)
# =========================================================
export RAG_ENABLED=false
run_batch 4011 4012 4013

echo "[$(date)] Full run finished."

