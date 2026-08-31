import argparse
import json
import os

from tqdm import tqdm

from bef4llm.definitions import Folder
from bef4llm.llm_comparison import prepare_datasets, quality_check
from bef4llm.llm_comparison.generate_bpmns import Benchmark
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.statistical_tests.pragmatic_quality_analysis import run_pragmatic_statistical_tests
from bef4llm.statistical_tests.semantic_quality_analysis import run_semantic_statistical_tests
from bef4llm.statistical_tests.statistical_tests_datasets import (
    generate_data_for_statistical_tests,
)
from bef4llm.statistical_tests.syntactic_quality_analysis import run_syntactic_statistical_tests
from bef4llm.statistical_tests.tables import (
    table7,
    table8,
    table9,
    table10,
    table11,
    table12,
    table13,
    table14,
)

# Each table's generator function, and whether it needs the LLM list as well
# as the run list. Used to collapse the long table7..table14 dispatch below
# into one lookup instead of one elif branch per table.
TABLE_GENERATORS = {
    "table7": (table7.generate_table7, False),
    "table8": (table8.generate_table8, True),
    "table9": (table9.generate_table9, False),
    "table10": (table10.generate_table10, False),
    "table11": (table11.generate_table11, False),
    "table12": (table12.generate_table12, True),
    "table13": (table13.generate_table13, True),
    "table14": (table14.generate_table14, True),
}


def get_run_config_path(run):
    """Return the path to the run_config.json file for a given run.

    Args:
        run: Run/iteration identifier.
    """
    return os.path.join(get_folder_path(Folder.DATA), f"llm_run{run}", "run_config.json")


def save_run_config(run, llms, datasets, dataset_mode, rag_config=None, timeout_config=None):
    """Save the configuration for a benchmark run, or validate it against an existing one.

    Args:
        run: Run/iteration identifier.
        llms: List of LLM names used in this run.
        datasets: Dataset dict (only stored when dataset_mode == "general").
        dataset_mode: Mode used to select datasets.
        rag_config: Optional RAG configuration dict.
        timeout_config: Optional timeout configuration dict.
    """
    path = get_run_config_path(run)
    config = {
        "llms": llms,
        "dataset_mode": dataset_mode,
        "status": "unfinished",
    }

    if dataset_mode == "general":
        config["datasets"] = list(datasets.keys())

    if rag_config is not None:
        config["rag"] = rag_config

    if timeout_config is not None:
        config["timeout"] = timeout_config

    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)

        core_existing = {
            "llms": existing.get("llms"),
            "datasets": existing.get("datasets"),
            "dataset_mode": existing.get("dataset_mode"),
            "rag": existing.get("rag"),
            "timeout": existing.get("timeout"),
        }
        core_new = {
            "llms": config["llms"],
            "datasets": config.get("datasets"),
            "dataset_mode": config["dataset_mode"],
            "rag": config.get("rag"),
            "timeout": config.get("timeout"),
        }

        if core_existing != core_new:
            raise RuntimeError(
                f"Run {run} already exists with different configuration.\n"
                f"Existing: {core_existing}\n"
                f"New: {core_new}"
            )
        print("[DEBUG] Existing run config matches.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[DEBUG] Run config saved at {path}")


def update_run_status(run, status):
    """Update the status field in a run's config.json file.

    Args:
        run: Run/iteration identifier.
        status: New status string to write (e.g. "DONE", "FAILED").
    """
    path = get_run_config_path(run)

    if not os.path.exists(path):
        raise RuntimeError("Run config not found")

    with open(path, "r") as f:
        config = json.load(f)

    config["status"] = status

    with open(path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"[DEBUG] Run {run} status updated to {status}")


def get_llms():
    """Read and parse the LLMS environment variable into a list of model names.

    Returns:
        List of LLM name strings.
    """
    llms_env = os.getenv("LLMS")

    if not llms_env:
        raise RuntimeError(
            "LLMS is not set in .env\n"
            "Example:\n"
            "LLMS=llama3.1:latest,phi3:latest"
        )

    llm_list = [model_name.strip() for model_name in llms_env.split(",") if model_name.strip()]

    if not llm_list:
        raise RuntimeError("LLMS is empty after parsing.")

    print(f"[DEBUG] Using LLMs: {llm_list}")

    return llm_list


def get_runs():
    """Read and parse the STATISTICAL_RUNS environment variable into a list of run ids.

    Returns:
        List of run id strings.
    """
    runs_env = os.getenv("STATISTICAL_RUNS")

    run_list = [run_id.strip() for run_id in runs_env.split(",") if run_id.strip()]

    if not run_list:
        raise RuntimeError("STATISTICAL_RUNS is empty")

    print(f"[DEBUG] Using runs: {run_list}")
    return run_list


def generate_bpmn(run):
    """Generate BPMN models for every configured LLM and dataset for a given run.

    Reads LLMs and RAG settings from environment variables, saves/validates
    the run config, then runs the benchmark for each (LLM, dataset) pair.
    Marks the run status as "DONE" or "FAILED" in its config file.

    Args:
        run: Run/iteration identifier.
    """
    llms = get_llms()
    datasets = quality_check.get_datasets_config()

    dataset_mode = os.getenv("DATASET_MODE")

    timeout_config = {
        "load_timeout": int(os.getenv("LOAD_TIMEOUT", "1800")),
        "generate_timeout": int(os.getenv("GENERATE_TIMEOUT", "300")),
    }

    rag_config = {
        "enabled": os.getenv("RAG_ENABLED", "false").lower() == "true",
        "dir": os.getenv("RAG_DIR"),
        "mode": os.getenv("RAG_MODE", "documents").strip().lower(),
        "top_k": int(os.getenv("RAG_TOP_K", 3)),
        "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", 500)),
        "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", 100)),
    }

    save_run_config(
        run,
        llms,
        datasets,
        dataset_mode,
        rag_config,
        timeout_config,
    )

    try:
        print(
            "Number of textual descriptions:",
            sum(len(datasets[dataset_name].keys()) for dataset_name in datasets),
        )

        for llm in llms:
            print(f"\n[INFO] Running LLM: {llm}")

            os.makedirs(
                os.path.join(get_folder_path(Folder.DATA), f"llm_run{run}", llm),
                exist_ok=True,
            )

            for dataset in tqdm(datasets):
                target_dir = os.path.join(
                    get_folder_path(Folder.DATA), f"llm_run{run}", llm, dataset,
                )

                print("Start test with dataset:", dataset)

                benchmark = Benchmark(
                    datasets[dataset],
                    llm=llm,
                    dataset_name=dataset,
                    load_timeout=timeout_config["load_timeout"],
                    generate_timeout=timeout_config["generate_timeout"],
                )

                benchmark.model_processes(
                    target_dir=target_dir,
                    rag_config=rag_config,
                )

        update_run_status(run, "DONE")

    except Exception as e:
        update_run_status(run, "FAILED")
        raise e


def check_quality_llms(run, evaluation):
    """Analyze the quality of the LLM-generated BPMNs for a run.

    Results are saved to a CSV file `llm_results_run{run}_{evaluation}.csv`
    in the folder `llm_run{run}`.

    Args:
        run: Iteration number.
        evaluation: "quality_group_score", "detail", or "metrics" — see the
            quality_check module.

    Returns:
        Whatever quality_check.run_quality_check_for_run returns.
    """
    return quality_check.run_quality_check_for_run(run=run, evaluation=evaluation)


def human_expert_comparison(run, evaluation):
    """Compare LLM-generated BPMN quality against the human-expert baseline.

    Computes quality metrics for the human-expert dataset and for the
    LLM-generated BPMNs of a given run, saving each to its own CSV file.

    Args:
        run: Iteration number.
        evaluation: Granularity of the analysis:
            "quality_group_score" - scores for syntactic, pragmatic, semantic
                quality, and validity.
            "detail" - subgroup scores for each quality dimension.
            "metrics" - result for each individual metric.
    """
    datasets = dict()
    datasets["experts"] = prepare_datasets.prepare_experts_comparison()

    human_dir = os.path.join(
        get_folder_path(Folder.DATA_HUMAN_COMPARISON), "Expert_BPMN", "experts_modelling",
    )
    human_csv = os.path.join(
        get_folder_path(Folder.DATA_HUMAN_COMPARISON),
        "Expert_BPMN",
        f"expert_results_{evaluation}.csv",
    )
    human_quality_df = quality_check.compute_overall_quality_llms(
        datasets=datasets,
        test_llm_dir=human_dir,
        target_file=human_csv,
        analyse_method=evaluation,
        is_human=True,
    )

    llm_dir = os.path.join(get_folder_path(Folder.DATA), f"llm_run{run}")
    llm_csv = os.path.join(
        get_folder_path(Folder.DATA), f"llm_run{run}", f"llm_results_run{run}_{evaluation}.csv",
    )
    llm_quality_df = quality_check.compute_overall_quality_llms(
        datasets=datasets,
        test_llm_dir=llm_dir,
        target_file=llm_csv,
        analyse_method=evaluation,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-function script")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate_bpmn")
    gen_parser.add_argument("run", type=str)

    quality_parser = subparsers.add_parser("check_quality_llms")
    quality_parser.add_argument("run", type=str)
    quality_parser.add_argument("evaluation", type=str)

    human_parser = subparsers.add_parser("human_expert_comparison")
    human_parser.add_argument("run", type=str)
    human_parser.add_argument("evaluation", type=str)

    stats_data_parser = subparsers.add_parser("statistical_datasets")

    stats_parser = subparsers.add_parser("statistical_tests")
    stats_parser.add_argument("metric", type=str)

    args = parser.parse_args()

    if args.command == "generate_bpmn":
        generate_bpmn(run=args.run)
    elif args.command == "check_quality_llms":
        check_quality_llms(run=args.run, evaluation=args.evaluation)
    elif args.command == "human_expert_comparison":
        human_expert_comparison(run=args.run, evaluation=args.evaluation)
    elif args.command == "statistical_datasets":
        llms = get_llms()
        datasets = quality_check.get_datasets_config()
        runs = get_runs()
        generate_data_for_statistical_tests(llms, datasets, runs)
    elif args.command == "statistical_tests":
        runs = get_runs()
        if args.metric == "syntactic":
            run_syntactic_statistical_tests(runs)
        elif args.metric == "pragmatic":
            run_pragmatic_statistical_tests(runs)
        elif args.metric == "semantic":
            run_semantic_statistical_tests(runs)
        elif args.metric == "all":
            run_syntactic_statistical_tests(runs)
            run_pragmatic_statistical_tests(runs)
            run_semantic_statistical_tests(runs)
        elif args.metric == "tables":
            llms = get_llms()
            for generate_table, needs_llms in TABLE_GENERATORS.values():
                generate_table(llms, runs) if needs_llms else generate_table(runs)
        elif args.metric in TABLE_GENERATORS:
            generate_table, needs_llms = TABLE_GENERATORS[args.metric]
            if needs_llms:
                llms = get_llms()
                generate_table(llms, runs)
            else:
                generate_table(runs)
