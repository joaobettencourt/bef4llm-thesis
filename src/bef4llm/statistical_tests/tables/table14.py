import os

import pandas as pd

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path, look_for_directory
from bef4llm.statistical_tests.tables.common import save_table


def count_expert_dataset_samples():
    """
    Number of textual descriptions in the human-comparison subset, counted
    dynamically from Expert_BPMN/textual_descriptions (instead of hardcoding 9).
    """
    dir_path = look_for_directory("Expert_BPMN")
    descriptions_dir = os.path.join(dir_path, "textual_descriptions")
    return len([
        f for f in os.listdir(descriptions_dir)
        if f.endswith(".txt") and not f.startswith(".")
    ])


def _row_for_llm(llm, runs, group_dir):
    """
    Reads data/statistical_datasets/llm_metric_results_run<run>/<llm>.csv for
    every run, pools all rows together (same pooling behaviour generate_table8
    gets "for free" by reading already-pooled CSVs), and returns a Table 14 row.
    """
    syn_scores, prag_scores, sem_scores = [], [], []
    n_valid_per_run, n_total_per_run = [], []

    for run in runs:
        path = f"{get_folder_path(Folder.DATA)}/statistical_datasets/llm_metric_results_run{run}/{llm}.csv"
        if not os.path.exists(path):
            print(f"[WARNING] {llm}: missing file for run {run} ({path}). Skipping this run.")
            continue

        df = pd.read_csv(path, sep=";")
        n_total_per_run.append(len(df))
        n_valid_per_run.append((df["status"] == "valid").sum())

        valid_df = df[df["status"] == "valid"]
        syn_scores.extend(valid_df["syntactic quality"].dropna().tolist())
        prag_scores.extend(valid_df["pragmatic quality"].dropna().tolist())
        sem_scores.extend(valid_df["semantic quality"].dropna().tolist())

    if not n_total_per_run:
        print(f"[WARNING] Skipping {llm}: no run files found at all.")
        return None

    avbm = sum(n_valid_per_run) / len(n_valid_per_run)
    avg_total = sum(n_total_per_run) / len(n_total_per_run)
    q_val = avbm / avg_total if avg_total else None

    if not syn_scores or not prag_scores or not sem_scores:
        print(f"[WARNING] Skipping {llm}: no valid rows with complete quality scores.")
        return None

    q_syn = sum(syn_scores) / len(syn_scores)
    q_prag = sum(prag_scores) / len(prag_scores)
    q_sem = sum(sem_scores) / len(sem_scores)
    q_qual = (q_syn + q_prag + q_sem) / 3
    q_total = (q_syn + q_prag + q_sem + q_val) / 4

    return {
        "LLM": llm,
        "Q_val": round(q_val, 4),
        "AVBM": round(avbm, 1),
        "Q_syn": round(q_syn, 4),
        "Q_prag": round(q_prag, 4),
        "Q_sem": round(q_sem, 4),
        "Q_qual": round(q_qual, 4),
        "Q_total": round(q_total, 4),
    }


def _row_for_humans():
    """
    Builds the "human experts" row directly from the already-computed
    expert_results_quality_group_score.csv (MEAN row) -- no need to
    recompute anything from the raw .bpmn files.
    """
    dir_path = look_for_directory("Expert_BPMN")
    path = os.path.join(dir_path, "expert_results_quality_group_score.csv")

    df = pd.read_csv(path, sep=";", index_col=0)
    mean_row = df[df["llm"] == "MEAN"]
    if mean_row.empty:
        raise ValueError(f"No 'MEAN' row found in {path}.")
    mean_row = mean_row.iloc[0]

    q_syn = mean_row["syntactic quality"]
    q_prag = mean_row["pragmatic quality"]
    q_sem = mean_row["semantic quality"]
    q_val = mean_row["validity"]

    num_descriptions = count_expert_dataset_samples()
    avbm = q_val * num_descriptions

    q_qual = (q_syn + q_prag + q_sem) / 3
    q_total = (q_syn + q_prag + q_sem + q_val) / 4

    return {
        "LLM": "human experts",
        "Q_val": round(q_val, 4),
        "AVBM": round(avbm, 1),
        "Q_syn": round(q_syn, 4),
        "Q_prag": round(q_prag, 4),
        "Q_sem": round(q_sem, 4),
        "Q_qual": round(q_qual, 4),
        "Q_total": round(q_total, 4),
    }


def generate_table14(llms, runs):
    """
    Builds Table 14 (validity, AVBM, per-dimension quality scores, and
    aggregate totals per LLM, plus a human-experts row) on the 9-description
    human-comparison subset, and saves it as table14.csv in the same
    statistical_tests_group_<runs>/tables directory used for Table 7/8.
    """
    runs_suffix = "_".join(str(r) for r in runs)
    group_dir = f"{get_folder_path(Folder.DATA)}/statistical_tests/statistical_tests_group_{runs_suffix}"

    rows = []
    for llm in llms:
        row = _row_for_llm(llm, runs, group_dir)
        if row is not None:
            rows.append(row)

    rows.append(_row_for_humans())

    table14_df = pd.DataFrame(rows)

    print("\n### Table 14: Comparison of Human Experts and LLM capabilities "
          "(Subset for the Human Evaluation) ###")
    print("=" * 70)
    print(table14_df.to_string(index=False))

    outdir = f"{group_dir}/tables"
    save_table(table14_df, "table14", outdir)

    return table14_df