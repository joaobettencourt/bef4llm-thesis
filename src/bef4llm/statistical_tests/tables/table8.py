import pandas as pd

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.llm_comparison import quality_check
from bef4llm.statistical_tests.tables.common import save_table

# Minimum fraction of valid BPMN-XMLs (averaged across runs) an LLM must
# produce to be included in Table 8. The paper used a fixed count of 30
# out of 105 samples (~30%); expressed as a ratio here so it scales
# automatically if the sample size or run count ever changes.
MIN_VALID_RATIO = 0.10


def generate_table8(llms, runs):
    """
    Builds Table 8 (validity, AVBM, per-dimension quality scores, and
    aggregate totals per LLM) and saves it as table8.csv, in the same
    statistical_tests_group_<runs>/tables directory used for Table 7.
    """
    runs_suffix = "_".join(str(r) for r in runs)
    group_dir = f"{get_folder_path(Folder.DATA)}/statistical_tests/statistical_tests_group_{runs_suffix}"

    # ------------------------------------------------------------------
    # 1. Q_syn, Q_prag, Q_sem: column means from the pooled quality CSVs
    #    (same files Table 7 already reads)
    # ------------------------------------------------------------------
    metric_files = {
        "Q_syn": "syntactic quality.csv",
        "Q_prag": "pragmatic quality.csv",
        "Q_sem": "semantic quality.csv",
    }

    quality_means = {}  # {llm: {"Q_syn": ..., "Q_prag": ..., "Q_sem": ...}}
    for col_name, filename in metric_files.items():
        path = f"{group_dir}/{filename}"
        df = pd.read_csv(path, sep=";", index_col=0)
        means = df.mean(axis=0, skipna=True)
        for llm, value in means.items():
            quality_means.setdefault(llm, {})[col_name] = value

    # ------------------------------------------------------------------
    # 2. Q_val and AVBM: run the quality check per run, then average the
    #    validity ratio and raw valid counts per LLM across all runs
    # ------------------------------------------------------------------
    per_run_frames = []
    for run in runs:
        df_run = quality_check.run_quality_check_for_run(run=run, evaluation="quality_group_score")
        df_run = df_run.set_index("llm")
        per_run_frames.append(df_run)

    validity_rows = []
    for llm in llms:
        run_num_valid, run_totals = [], []
        for df_run in per_run_frames:
            if llm not in df_run.index:
                continue
            row = df_run.loc[llm]
            if pd.notna(row.get("num_valid")):
                run_num_valid.append(row["num_valid"])
                run_totals.append(row.get("total"))

        avbm = (sum(run_num_valid) / len(run_num_valid)) if run_num_valid else None
        avg_total = (sum(run_totals) / len(run_totals)) if run_totals else None
        q_val = (avbm / avg_total) if (avbm is not None and avg_total) else None

        validity_rows.append({
            "llm": llm,
            "Q_val": q_val,
            "AVBM": avbm,
            "_avg_total_models": avg_total,
        })

    validity_df = pd.DataFrame(validity_rows).set_index("llm")

    # ------------------------------------------------------------------
    # 3. Merge, compute Q_qual / Q_total, apply the AVBM ratio filter
    # ------------------------------------------------------------------
    rows = []
    for llm in llms:
        q = quality_means.get(llm, {})
        v = validity_df.loc[llm] if llm in validity_df.index else None

        q_syn, q_prag, q_sem = q.get("Q_syn"), q.get("Q_prag"), q.get("Q_sem")
        q_val = v["Q_val"] if v is not None else None
        avbm = v["AVBM"] if v is not None else None
        avg_total = v["_avg_total_models"] if v is not None else None

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