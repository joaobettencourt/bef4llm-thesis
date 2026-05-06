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


def save_run_config(run, llms, datasets, dataset_mode):
    path = get_run_config_path(run)

    config = {
        "llms": llms,
        "datasets": list(datasets.keys()),
        "dataset_mode": dataset_mode,
        "status": "unfinished"
    }

    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)

        if existing["llms"] != config["llms"] or existing["datasets"] != config["datasets"]:
            raise RuntimeError(
                f"Run {run} already exists with different configuration.\n"
                f"Existing: {existing}\n"
                f"New: {config}"
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
            raise RuntimeError(
                "DATASETS must be set when DATASET_MODE=general"
            )

        dataset_names = [d.strip() for d in datasets_env.split(",") if d.strip()]

        allowed = {"camunda", "bpmn_and_text", "lre_new", "lre_old","small-camunda"}

        invalid = [d for d in dataset_names if d not in allowed]
        if invalid:
            raise ValueError(f"Invalid datasets in DATASETS: {invalid}")

        datasets = {}

        for name in dataset_names:
            if name == "camunda":
                datasets[name] = prepare_datasets.prepare_camunda()
            if name == "small-camunda":
                datasets[name] = prepare_datasets.prepare_small_camunda()
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

"""
def generate_bpmn(run):

    automates the process of generating BPMNs with multiple LLMs.
    All BPMNs are saved in the folder llm_runx, with x being the iteration of the experiment, and sorted by LLM and dataset.
    Each BPMN is saved in a separate BPMN-XML file.

    Prameters
    run: int
        indicates the number of the iteration

    llms = get_llms()
    datasets = get_datasets_config()

    print("Number of textual descriptions: ", sum([len(datasets[d].keys()) for d in datasets]))

    for llm in llms:
        print(f"\n[INFO] Running LLM: {llm}")
        if not os.path.isdir(f"{get_folder_path(Folder.DATA)}/llm_run{run}/{llm}"):
            os.makedirs(f"{get_folder_path(Folder.DATA)}/llm_run{run}/{llm}")

        not_modelled_models = []

        for dataset in tqdm(datasets):
            target_dir = f"{get_folder_path(Folder.DATA)}/llm_run{run}/{llm}/{dataset}"
            print("Start test with dataset:", dataset)
            benchmark = Benchmark(datasets[dataset], llm=llm)
            not_modelled = benchmark.model_processes(target_dir)
            not_modelled_models.extend(not_modelled)

        print("Due to a timeout the following models are not modelled or the invalid model is not corrected:",
              not_modelled_models)

        #with open(f"{get_folder_path(Folder.DATA)}/test_llm_temp_01_run{run}/{llm}/not_modelled.txt", "w") as f:
        #    f.write(str(not_modelled_models))
"""

def generate_bpmn(run):
    llms = get_llms()
    datasets = get_datasets_config()
    dataset_mode = os.getenv("DATASET_MODE")

    save_run_config(run, llms, datasets, dataset_mode)

    try:
        print("Number of textual descriptions: ", sum([len(datasets[d].keys()) for d in datasets]))

        for llm in llms:
            print(f"\n[INFO] Running LLM: {llm}")
            os.makedirs(f"{get_folder_path(Folder.DATA)}/llm_run{run}/{llm}", exist_ok=True)

            not_modelled_models = []

            for dataset in tqdm(datasets):
                target_dir = f"{get_folder_path(Folder.DATA)}/llm_run{run}/{llm}/{dataset}"
                print("Start test with dataset:", dataset)

                benchmark = Benchmark(datasets[dataset], llm=llm)
                not_modelled = benchmark.model_processes(target_dir)
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
    datasets = dict()
    datasets["camunda"] = prepare_datasets.prepare_camunda()
    datasets["small-camunda"] = prepare_datasets.prepare_small_camunda()
    datasets["bpmn_and_text"] = prepare_datasets.prepare_text_and_bpmn()
    datasets["lre_new"] = prepare_datasets.prepare_lre_new()
    datasets["lre_old"] = prepare_datasets.prepare_lre_old()

    df = quality_check.compute_overall_quality_llms(datasets=datasets,
                                                    test_llm_dir=f"{get_folder_path(Folder.DATA)}/llm_run{run}",
                                                    target_file=f"{get_folder_path(Folder.DATA)}//llm_run{run}/llm_results_run{run}_{evaluation}.csv",
                                                    analyse_method=evaluation)

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

