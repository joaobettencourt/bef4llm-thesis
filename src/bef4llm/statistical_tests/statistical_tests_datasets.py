import os
import pandas as pd

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.llm_comparison import quality_check


def calcualte_metric_results_per_bpmn_for_each_LLM(llms, datasets, runs):
    """
    Generates the files containing the metric results for each BPMN.
    For each LLM one file is generated, and saved in the folder
    llm_metric_results_run{run} as file {llm}.csv.
    """
    for llm in llms:
        for run in runs:
            output_dir = f"{get_folder_path(Folder.DATA)}/statistical_datasets/llm_metric_results_run{run}"
            os.makedirs(output_dir, exist_ok=True)

            quality_check.get_metric_results_per_process_model(
                datasets=datasets,
                llm_dir=f"{get_folder_path(Folder.DATA)}/llm_run{run}/{llm}",
                analyse_method="quality_group_score",
                run=run,
                target_file=f"{output_dir}/{llm}.csv"
            )


def generate_data_for_statistical_tests(llms, datasets, runs):
    """
    Generate the pivoted per-metric CSVs (bpmn x llm, averaged across runs)
    used by the Skillings-Mack tests (Table 7). Every expected BPMN gets a
    row, even if it's invalid/not_generated for every LLM (in which case
    its row is all NaN).
    """
    base_path = f"{get_folder_path(Folder.DATA)}/statistical_datasets"
    os.makedirs(base_path, exist_ok=True)

    # 1. Generate/refresh the per-model results CSVs (one per llm/run)
    calcualte_metric_results_per_bpmn_for_each_LLM(llms=llms, datasets=datasets, runs=runs)

    # 2. Load and stack all per-model CSVs into one long dataframe
    metrics = ["syntactic quality", "pragmatic quality", "semantic quality"]
    frames = []
    for run in runs:
        run_path = f"{base_path}/llm_metric_results_run{run}"
        if not os.path.isdir(run_path):
            print(f"[WARNING] Missing folder: {run_path}")
            continue

        for file in os.listdir(run_path):
            if not file.endswith(".csv"):
                continue
            llm = file.replace(".csv", "")
            df = pd.read_csv(f"{run_path}/{file}", sep=";")
            df["llm"] = llm
            frames.append(df[["process model", "run", "llm", *metrics]])

    all_results = pd.concat(frames, ignore_index=True)

    # long format: one row per (bpmn, run, llm, metric)
    long_df = all_results.melt(
        id_vars=["process model", "run", "llm"],
        value_vars=metrics,
        var_name="metric",
        value_name="value"
    ).rename(columns={"process model": "bpmn"})

    print(f"[DEBUG] Total rows collected: {len(long_df)}")
    print("[DEBUG] Non-null values per metric:")
    print(long_df.groupby("metric")["value"].apply(lambda x: x.notna().sum()))

    # 3. Pivot per metric (bpmn x llm, averaged across runs) and save.
    #    dropna=False keeps every bpmn as a row and every llm as a column,
    #    even if a whole row/column ends up all-NaN.
    runs_suffix = "_".join(str(r) for r in runs)
    output_dir = f"{get_folder_path(Folder.DATA)}/statistical_tests/statistical_tests_group_{runs_suffix}"
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        pivot = long_df[long_df["metric"] == metric].pivot_table(
            index="bpmn", columns="llm", values="value", aggfunc="mean", dropna=False
        ).reset_index()

        pivot.to_csv(f"{output_dir}/{metric}.csv", sep=";", index=False)
        print(f"[INFO] Created {metric}.csv")