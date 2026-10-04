import os
import pandas as pd

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.statistical_tests.tables.common import save_table

# Minimum fraction of valid BPMN-XMLs (averaged across runs) an LLM must
# produce to be included in Table 8. The paper used a fixed count of 30
# out of 105 samples (~30%); expressed as a ratio here so it scales
# automatically if the sample size or run count ever changes.
MIN_VALID_RATIO = 0.01


def _load_per_model_results(llm, run):
    """
    Reads the per-model results CSV produced by
    get_metric_results_per_process_model for a given llm/run.
    Returns None if the file doesn't exist yet.
    """
    path = f"{get_folder_path(Folder.DATA)}/statistical_datasets/llm_metric_results_run{run}/{llm}.csv"
    if not os.path.isfile(path):
        print(f"[WARNING] Missing per-model results file: {path}")
        return None
    return pd.read_csv(path, sep=";")


def generate_table8(llms, runs):
    """
    Builds Table 8 (validity, AVBM, per-dimension quality scores, and
    aggregate totals per LLM) and saves it as table8.csv, in the same
    statistical_tests_group_<runs>/tables directory used for Table 7.

    Everything is derived from the per-model results CSVs (one row per
    expected process model, with a 'status' column of valid/invalid/
    not_generated) produced by the 'statistical_datasets' step — no
    separate recomputation against the raw .bpmn files is needed here.
    """
    runs_suffix = "_".join(str(r) for r in runs)
    group_dir = f"{get_folder_path(Folder.DATA)}/statistical_tests/statistical_tests_group_{runs_suffix}"

    per_llm_run_stats = {}  # {llm: {"num_valid": [...], "total": [...], "Q_syn": [...], "Q_prag": [...], "Q_sem": [...]}}

    for llm in llms:
        stats = {
            "num_valid": [], "num_invalid": [], "num_not_generated": [],
            "total": [], "Q_syn": [], "Q_prag": [], "Q_sem": [],
        }
        for run in runs:
            df = _load_per_model_results(llm, run)
            if df is None:
                continue

            total = len(df)
            num_valid = (df["status"] == "valid").sum()
            num_invalid = (df["status"] == "invalid").sum()
            num_not_generated = (df["status"] == "not_generated").sum()

            stats["total"].append(total)
            stats["num_valid"].append(num_valid)
            stats["num_invalid"].append(num_invalid)
            stats["num_not_generated"].append(num_not_generated)
            stats["Q_syn"].append(df["syntactic quality"].mean(skipna=True))
            stats["Q_prag"].append(df["pragmatic quality"].mean(skipna=True))
            stats["Q_sem"].append(df["semantic quality"].mean(skipna=True))

        per_llm_run_stats[llm] = stats

    rows = []
    for llm in llms:
        stats = per_llm_run_stats[llm]

        if not stats["total"]:
            print(f"[WARNING] Skipping {llm}: no per-model data found for any run.")
            continue

        avbm = sum(stats["num_valid"]) / len(stats["num_valid"])
        avim = sum(stats["num_invalid"]) / len(stats["num_invalid"])
        avng = sum(stats["num_not_generated"]) / len(stats["num_not_generated"])
        avg_total = sum(stats["total"]) / len(stats["total"])
        q_val = avbm / avg_total if avg_total else None

        q_syn = pd.Series(stats["Q_syn"]).mean(skipna=True)
        q_prag = pd.Series(stats["Q_prag"]).mean(skipna=True)
        q_sem = pd.Series(stats["Q_sem"]).mean(skipna=True)

        values = (q_syn, q_prag, q_sem, q_val, avbm)
        if any(x is None or pd.isna(x) for x in values):
            print(f"[WARNING] Skipping {llm}: missing data for Table 8.")
            continue

        cutoff = MIN_VALID_RATIO * avg_total if avg_total else 0
        if avbm < cutoff:
            print(f"[INFO] Excluding {llm}: AVBM={avbm:.1f} below {MIN_VALID_RATIO:.0%} threshold ({cutoff:.1f}).")
            continue

        q_qual = (q_syn + q_prag + q_sem) / 3
        q_total = (q_syn + q_prag + q_sem + q_val) / 4

        rows.append({
            "LLM": llm,
            "Q_val": round(q_val, 4),
            "AVBM": round(avbm, 1),
            "AVIM": round(avim, 1),
            "AVNG": round(avng, 1),
            "Q_syn": round(q_syn, 4),
            "Q_prag": round(q_prag, 4),
            "Q_sem": round(q_sem, 4),
            "Q_qual": round(q_qual, 4),
            "Q_total": round(q_total, 4),
        })

    table8_df = pd.DataFrame(rows)

    print("\n### Table 8: Results of the first experiment ###")
    print("=" * 70)
    print(table8_df.to_string(index=False))

    outdir = f"{group_dir}/tables"
    save_table(table8_df, "table8", outdir)

    return table8_df