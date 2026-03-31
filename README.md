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

Copy the example environment file and update it with your server and models:

```
cp .env.example .env
```

Edit `.env`:

- OLLAMA_HOST: URL of your Ollama server  
- LLMS: Comma-separated list of models to use

These settings are automatically used by the scripts inside the container.

---

### Docker

Build the Docker image:

```
docker build -t bef4llm-docker .
```

Run the container interactively with your `.env` file. The `src` and `data` folders are mounted into the container (changes are reflected both on the host and inside the container). The container is automatically removed after exit:

```
docker run --rm -it \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/data:/app/data \
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

**Note:** The README has been fully verified and tested up to this point. All subsequent sections have been adapted from the old README with the help of generative AI and require further verification.

---

### Comparison to human experts

BPMNs from human experts can be compared by generating expert BPMNs:

```
python src/compare_llms.py generate_bpmn --expert_dataset=True
```

Then assess the comparison with:

```
check_quality_expert_dataset
```

The expert dataset is in the folder `data_human_comparison`.

---

### Statistical tests

Statistical tests are in the folder `statistical_tests/`, available for syntactic, pragmatic, and semantic quality dimensions.  

To run a test, e.g., for syntactic quality:

```
python statistical_tests/syntactic_quality_analysis.py
```

A dataset is automatically created at the beginning of each test. Each file contains scores of BPMNs for each run and each LLM. Invalid or missing BPMNs are indicated by `none` or `NaN`.