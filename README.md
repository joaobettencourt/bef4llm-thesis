# BEF4LLM

This repository is an adaptation of the original [BEF4LLM](https://gitlab-iwi.dfki.de/lauer/bef4llm), which was designed to evaluate the ability of large language models (LLMs) to generate BPMN diagrams and compare their outputs to those of human experts. The original pipeline was built to analyze how well LLMs can generate BPMNs from textual descriptions using a dataset of 105 text-BPMN pairs, and to compare their performance against human experts using a smaller dataset of 9 text-model pairs.

The original study is described in the paper:  
*Assessing the Business Process Modeling Competences of Large Language Models*

In this adaptation, the framework has been modified to run in a Docker container, and the configurations have been simplified for easier setup and reproducibility, supporting extensions to investigate the impact of integrating retrieval-augmented generation (RAG) into LLM workflows as part of my master's thesis. It retains the original evaluation pipeline, allowing testing of multiple LLMs, analyzing generated BPMNs, and comparing their performance to expert-modeled BPMNs.

**Note:** These instructions assume the framework is run in Docker, but the Ollama server should be running externally (e.g., on your host machine or another server) and be accessible via the OLLAMA_HOST environment variable.

---

## Setup & Running

### Installation

The installation via pip is not yet tested, so we recommend cloning the project and working with it.  
All required packages can be installed by using the `requirements.txt`, with the most important packages being: 

```
- ollama
- pandas
- networkx
```

---

### Ollama

Ensure your Ollama server is running:

```
ollama serve
```

- Use `ollama list` to see which models are installed  
- Make sure the models you want to use are available in your Ollama instance  

---

### Configuration

Copy the example environment file and update it with your configuration:

```
cp .env.example .env
```

Edit `.env`:

- `OLLAMA_HOST`: URL of your Ollama server  
- `LLMS`: Comma-separated list of models to use  
- `DATASET_MODE`:
  - `general` → use selected datasets
  - `experts` → use only the expert dataset  
- `DATASETS` (only used if `DATASET_MODE=general`):
  - Comma-separated subset of:
    `camunda`, `bpmn_and_text`, `lre_new`, `lre_old`
- `RAG_ENABLED` and other RAG definitions (see `.env.example` for more detail)

---

### Docker

Build the Docker image:

```
docker build -t bef4llm-docker .
```

Run the container interactively with your `.env` file. The `src`, `data` and `data_human_comparison` folders are mounted into the container (changes are reflected both on the host and inside the container). The container is automatically removed after exit:


```
docker run --rm -it \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/data_human_comparison:/app/data_human_comparison \
  -v $(pwd)/rag:/app/rag \
  -v $(pwd)/scripts:/app/scripts \
  --env-file .env \
  bef4llm-docker \
  bash
```

---

## Structure and Usage of the Repository

This repository allows assessing the ability of LLMs to generate BPMNs, comparing them to human experts, and performing statistical analyses of the results.

### Assessing the abilities of LLMs to generate BPMNs

Two steps are necessary: generating the BPMNs and analyzing the generated BPMNs.

The parameter `run` is an identifier for a specific experiment. It determines the folder where results are stored (e.g., `data/llm_run1/`).

If you wish to use RAG, place your documents inside the `rag/` directory at the root of the project.
Each RAG dataset must be stored in its own subdirectory inside `rag/`.
Then, specify the subdirectory name in the `.env` file using the `RAG_DIR` variable.


**Step 1: Generation of BPMNs by LLMs**  

```
python src/compare_llms.py generate_bpmn 1
```

---

**Step 2: Assessment of Generated BPMNs**  

```
python src/compare_llms.py check_quality_llms 1 quality_group_score
```

Replace `quality_group_score` with the desired evaluation level:
- `quality_group_score` – overall scores for the four main categories (syntactic, pragmatic, semantic, validity)
- `detail` – subgroup scores within each quality dimension
- `metric_score` – individual metric values for fine-grained analysis

---

### Comparison to human experts

The repository includes a dataset of BPMNs modeled by human experts, located in the `data_human_comparison` folder. This dataset contains textual descriptions, ground truth BPMNs, and BPMNs created by multiple experts.

**Step 1: Generate BPMNs for the expert dataset (LLMs only)** 

Set `DATASET_MODE=experts` in `.env`, then run:

```
python src/compare_llms.py generate_bpmn 2
```

**Step 2: Compute evaluation metrics for both LLMs and human experts**

```
python src/compare_llms.py human_expert_comparison 2 quality_group_score
```

Replace `quality_group_score` with the desired evaluation level.

**Output**

- LLM results are saved under: `data/llm_runX/llm_results_runX_<evaluation>.csv`
- Human expert results are saved under: `data_human_comparison/Expert_BPMN/expert_results_<evaluation>.csv`

---

### Statistical tests

**Step 1: Generate datasets** 

First, the statistical datasets must be generated from the results of the configured runs and LLMs:

```
python src/compare_llms.py statistical_datasets
```

This step processes the outputs of all runs and LLMs and creates the datasets required for statistical analysis.
It produces:
- A folder per run inside `statistical_datasets`, containing one file per LLM
- A `statistical_tests_group_<runs>` folder inside `statistical_tests`, containing three files (one per metric), where each entry corresponds to a BPMN across runs

**Step 2: Run statistical tests** 

```
python src/compare_llms.py statistical_tests <metric>
```

Where `<metric>` can be:
- `syntactic` → syntactic quality analysis
- `pragmatic` → pragmatic quality analysis
- `semantic` → semantic quality analysis
- `all` → runs all statistical tests


## Command to build subset of the camunda dataset
```
./generate_camunda_subset.sh 2 \
        01-Dispatch-of-goods \
        02-Recourse
```

run outside docker
```
chmod +x scripts/*.sh
```