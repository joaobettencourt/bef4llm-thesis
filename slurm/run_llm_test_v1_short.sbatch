#!/bin/bash
#SBATCH --job-name=basic_test_short
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --qos=gpu075522026
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/users5/cpca075522026/jbettencourt/bef4llm-thesis/slurm/logs/%x_%j.out
#SBATCH --error=/users5/cpca075522026/jbettencourt/bef4llm-thesis/slurm/logs/%x_%j.err

set -e

# =========================================================
# Project root (do not rely on --chdir)
# =========================================================
PROJECT_ROOT="/users5/cpca075522026/jbettencourt/bef4llm-thesis"

cd "$PROJECT_ROOT"
echo "[$(date)] Working directory: $(pwd)"

# =========================================================
# 1. Start Ollama and wait until it is ready
# =========================================================
echo "[$(date)] Starting ollama serve..."
ollama serve > "$PROJECT_ROOT/slurm/logs/ollama_${SLURM_JOB_ID}.log" 2>&1 &
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
# 3. Build only the camunda_4 dataset
# =========================================================
cd scripts

if [ ! -d "../data/models/camunda_4" ]; then
    echo "[$(date)] Generating camunda_4 dataset..."
    ./generate_camunda_subset.sh 4 04-Self-service-restaurant
fi

cd ..

# =========================================================
# 4. Single run, single LLM, no RAG (smoke test)
# =========================================================
export OLLAMA_HOST=http://localhost:11434
export LLMS=qwen2.5:latest
export DATASET_MODE=general
export DATASETS=camunda_4
export RAG_ENABLED=false

RUN=9001
echo "[$(date)] Generating BPMNs for test run $RUN"

log_dir="data/llm_run${RUN}"
mkdir -p "$log_dir"

python src/compare_llms.py generate_bpmn "$RUN" \
    2>&1 | tee "$log_dir/run_${RUN}.log"

echo "[$(date)] Test run finished."
