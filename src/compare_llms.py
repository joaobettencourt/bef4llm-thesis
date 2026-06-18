import argparse
import re
import sys

from bef4llm.llm_comparison import prepare_datasets, quality_check
from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path

import os
from tqdm import tqdm
from bef4llm.llm_comparison.generate_bpmns import Benchmark

from bef4llm.statistical_tests.syntactic_quality_analysis import run_syntactic_statistical_tests
from bef4llm.statistical_tests.pragmatic_quality_analysis import run_pragmatic_statistical_tests
from bef4llm.statistical_tests.semantic_quality_analysis import run_semantic_statistical_tests

from bef4llm.statistical_tests.statistical_tests_datasets import (
    generate_data_for_statistical_tests
)

import json


def get_run_config_path(run):
    return f"{get_folder_path(Folder.DATA)}/llm_run{run}/run_config.json"


def save_run_config(run, llms, datasets, dataset_mode, rag_config=None):
    path = get_run_config_path(run)
    config = {
        "llms": llms,
        "dataset_mode": dataset_mode,
        "status": "unfinished"
    }

    # Only store datasets list if in general mode
    if dataset_mode == "general":
        config["datasets"] = list(datasets.keys())

    # Add RAG config if provided
    if rag_config is not None:
        config["rag"] = rag_config

    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)

        core_existing = {
            "llms": existing.get("llms"),
            "datasets": existing.get("datasets"),
            "dataset_mode": existing.get("dataset_mode"),
            "rag": existing.get("rag"),
        }
        core_new = {
            "llms": config["llms"],
            "datasets": config.get("datasets"),
            "dataset_mode": config["dataset_mode"],
            "rag": config.get("rag"),
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
    llms = os.getenv("LLMS")
    
    if not llms:
        raise RuntimeError(
            "LLMS is not set in .env\n"
            "Example:\n"
            "LLMS=llama3.1:latest,phi3:latest"
        )
    
    llm_list = [m.strip() for m in llms.split(",") if m.strip()]
    
    if not llm_list:
        raise RuntimeError("LLMS is empty after parsing.")
    
    print(f"[DEBUG] Using LLMs: {llm_list}")
    
    return llm_list

def get_datasets_config():
    mode = os.getenv("DATASET_MODE")
    if not mode:
        raise RuntimeError(
            "DATASET_MODE is not set in .env\n"
            "Options: general | experts"
        )
    mode = mode.strip().lower()

    if mode == "experts":
        print("[DEBUG] Using expert dataset only")
        return {"experts": prepare_datasets.prepare_experts_comparison()}

    elif mode == "general":
        datasets_env = os.getenv("DATASETS")
        if not datasets_env:
            raise RuntimeError("DATASETS must be set when DATASET_MODE=general")

        dataset_names = [d.strip() for d in datasets_env.split(",") if d.strip()]

        # Valida nomes: fixos + camunda_<int>
        fixed_allowed = {"camunda", "bpmn_and_text", "lre_new", "lre_old"}

        def is_valid(name):
            if name in fixed_allowed:
                return True
            parts = name.split("_")
            return len(parts) == 2 and parts[0] == "camunda" and parts[1].isdigit()

        invalid = [d for d in dataset_names if not is_valid(d)]
        if invalid:
            raise ValueError(f"Invalid datasets in DATASETS: {invalid}")

        datasets = {}
        for name in dataset_names:
            if name == "camunda":
                datasets[name] = prepare_datasets.prepare_camunda()
            elif name.startswith("camunda_") and name.split("_")[1].isdigit():
                version = int(name.split("_")[1])
                datasets[name] = prepare_datasets.prepare_camunda(version=version)
            elif name == "bpmn_and_text":
                datasets[name] = prepare_datasets.prepare_text_and_bpmn()
            elif name == "lre_new":
                datasets[name] = prepare_datasets.prepare_lre_new()
            elif name == "lre_old":
                datasets[name] = prepare_datasets.prepare_lre_old()

        print(f"[DEBUG] Using datasets: {list(datasets.keys())}")
        return datasets

    else:
        raise ValueError("DATASET_MODE must be 'general' or 'experts'")

def get_runs():
    runs = os.getenv("STATISTICAL_RUNS")

    run_list = [r.strip() for r in runs.split(",") if r.strip()]

    if not run_list:
        raise RuntimeError("STATISTICAL_RUNS is empty")

    print(f"[DEBUG] Using runs: {run_list}")
    return run_list


def generate_bpmn(run):
    llms = get_llms()
    datasets = get_datasets_config()

    dataset_mode = os.getenv("DATASET_MODE")

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
        rag_config
    )

    try:
        print(
            "Number of textual descriptions:",
            sum(len(datasets[d].keys()) for d in datasets)
        )

        for llm in llms:
            print(f"\n[INFO] Running LLM: {llm}")

            os.makedirs(
                f"{get_folder_path(Folder.DATA)}/llm_run{run}/{llm}",
                exist_ok=True
            )

            not_modelled_models = []

            for dataset in tqdm(datasets):
                target_dir = (
                    f"{get_folder_path(Folder.DATA)}/"
                    f"llm_run{run}/{llm}/{dataset}"
                )

                print("Start test with dataset:", dataset)

                benchmark = Benchmark(
                    datasets[dataset],
                    llm=llm,
                    dataset_name=dataset
                )

                not_modelled = benchmark.model_processes(
                    target_dir=target_dir,
                    rag_config=rag_config
                )

                not_modelled_models.extend(not_modelled)

        update_run_status(run, "DONE")

    except Exception as e:
        update_run_status(run, "FAILED")
        raise e

def check_quality_llms(run, evaluation):
    """
    Allows to analyze the quality of the LLM-generated BPMNs.
    The analysis results are saved in a csv file llm_results_runx_evaluation (x= iteration, evaluation = evaluation type) in the folder llm_runx
    run: int
        indicates the number of the iteration
    evaluation: string
        We allow for three types of granualrity to assess the quality of the LLM-generated BPMNs.
        quality_group_score: scores in the four groups sytactic, pragmatic and semantic quality, and validity
        detail: here the subgroup scores for the quality dimensions are calculated
        metrics: here each metric result is given
    expert_dataset:
        if true, the analysis is made for the expert comparison
    """
    datasets = get_datasets_config()

    # Also index datasets by the directory names used in LLM runs
    # e.g. if datasets has "camunda", also expose it as "camunda_1", "camunda_2", etc.
    # by scanning the actual subdirectories in the run folder
    llm_run_dir = f"{get_folder_path(Folder.DATA)}/llm_run{run}"
    for llm in os.listdir(llm_run_dir):
        llm_path = os.path.join(llm_run_dir, llm)
        if not os.path.isdir(llm_path) or llm.startswith("."):
            continue
        for dataset_dir in os.listdir(llm_path):
            if dataset_dir not in datasets:
                # try to resolve it to a known dataset
                parts = dataset_dir.split("_")
                if len(parts) == 2 and parts[0] == "camunda" and parts[1].isdigit():
                    version = int(parts[1])
                    datasets[dataset_dir] = prepare_datasets.prepare_camunda(version=version)

    df = quality_check.compute_overall_quality_llms(
        datasets=datasets,
        test_llm_dir=llm_run_dir,
        target_file=f"{get_folder_path(Folder.DATA)}/llm_run{run}/llm_results_run{run}_{evaluation}.csv",
        analyse_method=evaluation
    )

def human_expert_comparison(run, evaluation, expert_dataset=False):
    """
    Allows to analyze the quality of the LLM-generated BPMNs and compare the results to human experts.
    This is based on the dataset of the human experts.
    The analysis results are saved in a csv file llm_results_runx_evaluation (x= iteration, evaluation = evaluation type) in the folder llm_runx
    run: int
        indicates the number of the iteration
    evaluation: string
        We allow for three types of granualrity to assess the quality of the LLM-generated BPMNs.
        quality_group_score: scores in the four groups sytactic, pragmatic and semantic quality, and validity
        detail: here the subgroup scores for the quality dimensions are calculated
        metrics: here each metric result is given
    expert_dataset:
        if true, the analysis is made for the expert comparison
    """
    #datasets = dict()
    #datasets["experts"] = prepare_datasets.prepare_experts_comparison()

    # Compute metrics for the human expert BPMNs
    #df = quality_check.compute_overall_quality_llms(datasets=datasets,
    #                                                test_llm_dir=f"{get_folder_path(Folder.DATA)}/expert_comparison/human_experts",
    #                                                target_file=f"{get_folder_path(Folder.DATA)}/expert_comparison/human_experts/expert_results_{evaluation}.csv",
    #                                                analyse_method=evaluation)

    # Compute metrics for the LLM-generated BPMNs
    #df = quality_check.compute_overall_quality_llms(datasets=datasets,
    #                                                test_llm_dir=f"{get_folder_path(Folder.DATA)}/llm_run{run}",
    #                                                target_file=f"{get_folder_path(Folder.DATA)}//llm_run{run}/llm_results_run{run}_{evaluation}.csv",
    #                                                analyse_method=evaluation)

    datasets = dict()
    datasets["experts"] = prepare_datasets.prepare_experts_comparison()

    # Metrics for human experts
    human_dir = os.path.join(get_folder_path(Folder.DATA_HUMAN_COMPARISON), "Expert_BPMN", "experts_modelling")
    human_csv = os.path.join(get_folder_path(Folder.DATA_HUMAN_COMPARISON), "Expert_BPMN", f"expert_results_{evaluation}.csv")
    df = quality_check.compute_overall_quality_llms(
        datasets=datasets,
        test_llm_dir=human_dir,
        target_file=human_csv,
        analyse_method=evaluation,
        is_human=True
    )

    # Metrics for LLM-generated BPMNs
    llm_dir = os.path.join(get_folder_path(Folder.DATA), f"llm_run{run}")
    llm_csv = os.path.join(get_folder_path(Folder.DATA), f"llm_run{run}", f"llm_results_run{run}_{evaluation}.csv")
    df = quality_check.compute_overall_quality_llms(
        datasets=datasets,
        test_llm_dir=llm_dir,
        target_file=llm_csv,
        analyse_method=evaluation
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-function script")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # arguments for generating BPMNs
    gen_parser = subparsers.add_parser("generate_bpmn")
    gen_parser.add_argument("run", type=str)

    # arguments for quality check
    quality_parser = subparsers.add_parser("check_quality_llms")
    quality_parser.add_argument("run", type=str)
    quality_parser.add_argument("evaluation", type=str)

    # arguments for human expert comparison
    human_parser = subparsers.add_parser("human_expert_comparison")
    human_parser.add_argument("run", type=str)
    human_parser.add_argument("evaluation", type=str)

    # arguments for building the statistical datasets
    stats_data_parser = subparsers.add_parser("statistical_datasets")

    # arguments for statistical tests
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
        datasets = get_datasets_config()
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

