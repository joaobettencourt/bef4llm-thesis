import copy

from bef4llm.llm_comparison import prepare_datasets, quality_check
from bef4llm.definitions import *

import pandas as pd

def calcualte_metric_results_per_bpmn_for_each_LLM(llms, datasets, runs):
    """
    Generates the files containing the metric results for each BPMN.
    For each LLM one file is generated, and saved in the folder llm_metric_results_runx as file name_of_llm.csv

    Parameters
    -----------
    llms : list
        list of LLM tags
    datasets : dictionary
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
    Generate datasets for statistical tests.
    Creates one CSV per metric with all BPMNs across LLMs and runs.
    """

    base_path = f"{get_folder_path(Folder.DATA)}/statistical_datasets"
    os.makedirs(base_path, exist_ok=True)

    # ---------------------------------------
    # 1. Generate metric CSVs (per LLM/run)
    # ---------------------------------------
    calcualte_metric_results_per_bpmn_for_each_LLM(
        llms=llms,
        datasets=datasets,
        runs=runs
    )

    # ---------------------------------------
    # 2. Read generated CSVs
    # ---------------------------------------
    rows = []
    metrics = ["syntactic quality", "pragmatic quality", "semantic quality"]

    for run in runs:
        run_path = f"{base_path}/llm_metric_results_run{run}"

        if not os.path.isdir(run_path):
            print(f"[WARNING] Missing folder: {run_path}")
            continue

        for file in os.listdir(run_path):
            if not file.endswith(".csv"):
                continue

            llm = file.replace(".csv", "")
            file_path = f"{run_path}/{file}"

            print(f"[DEBUG] Processing {file} (run {run})")

            df = pd.read_csv(file_path, sep=";")
            print(f"[DEBUG] Columns in {file}: {list(df.columns)}")

            for _, row in df.iterrows():
                bpmn = row["process model"]

                if bpmn == "process model":
                    continue

                print(f"\n[DEBUG] Row BPMN: {bpmn}")

                for metric in metrics:
                    value = row.get(metric, None)

                    print(f"[DEBUG]   metric: '{metric}' -> value: {value}")

                    rows.append({
                        "bpmn": bpmn,
                        "run": run,
                        "llm": llm,
                        "metric": metric,
                        "value": value
                    })

    # ---------------------------------------
    # 3. Convert to DataFrame + save
    # ---------------------------------------
    runs_suffix = "_".join(str(r) for r in runs)
    output_dir = f"{get_folder_path(Folder.DATA)}/statistical_tests/statistical_tests_group_{runs_suffix}"
    #output_dir = f"{get_folder_path(Folder.DATA)}/Analysis Results/statistical_tests_group"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    print("\n[DEBUG] DataFrame head:")
    print(df.head())

    print("\n[DEBUG] Unique metrics in DF:")
    print(df["metric"].unique())

    print("\n[DEBUG] Unique LLMs in DF:")
    print(df["llm"].unique())

    print("\n[DEBUG] Non-null values per metric:")
    print(df.groupby("metric")["value"].apply(lambda x: x.notna().sum()))

    print(f"[DEBUG] Total rows collected: {len(df)}")

    for metric in metrics:
        df_metric = df[df["metric"] == metric]

        pivot = df_metric.pivot_table(
            index="bpmn",
            aggfunc="mean",
            columns="llm",
            values="value"
        ).reset_index()

        pivot.to_csv(f"{output_dir}/{metric}.csv", sep=";", index=False)

        print(f"[INFO] Created {metric}.csv")