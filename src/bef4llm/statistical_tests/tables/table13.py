import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.statistical_tests.tables.common import save_table
from bef4llm.statistical_tests.tables.table12 import split_family_and_setting, RAG_SETTINGS

# The three quality dimensions, and the CSV files they come from
# (same files already used by generate_table8 / generate_table12).
METRIC_FILES = {
    "p_syntactic": "syntactic quality.csv",
    "p_pragmatic": "pragmatic quality.csv",
    "p_semantic": "semantic quality.csv",
}

MIN_PAIRED_CASES = 10  # same threshold used in run_pairwise_wilcoxon
SIGNIFICANCE_ALPHA = 0.05


def format_p(p, adj):
    """
    Formats a p-value as 'n.s.' or as scientific notation with a
    significance marker, e.g. '2.75 × 10-9 ***', matching the paper's
    Table 13 style (*p<0.05, **p<0.01, ***p<0.001).
    """
    if pd.isna(p):
        return "n.s."
    if adj >= SIGNIFICANCE_ALPHA:
        return "n.s."
    mantissa, exponent = f"{adj:.2e}".split("e")
    formatted = f"{float(mantissa):.2f} × 10{int(exponent)}"
    if adj < 0.001:
        stars = "***"
    elif adj < 0.01:
        stars = "**"
    else:
        stars = "*"
    return f"{formatted} {stars}"


def generate_table13(llms, runs):
    """
    Builds Table 13 (Wilcoxon signed-rank tests contrasting the two RAG
    settings -- baseline vs examples_solution -- within each LLM family,
    Bonferroni-adjusted separately per quality dimension), analogous to the
    paper's Table 13 which contrasts small vs large checkpoints within each
    LLM family.

    Note: unlike Table 9/10/11, this does NOT reuse run_pairwise_wilcoxon,
    because that function tests every admissible pair of LLM configs (15 in
    your case), while here we only want the 1 relevant contrast per family
    (baseline vs examples_solution). It reuses the same wilcoxon test and
    the same Bonferroni style (p * n_tests, clipped to 1) though, just
    restricted to the family-relevant contrasts.

    Saved as table13.csv in statistical_tests_group_<runs>/tables.
    """
    runs_suffix = "_".join(str(r) for r in runs)
    group_dir = f"{get_folder_path(Folder.DATA)}/statistical_tests/statistical_tests_group_{runs_suffix}"

    if len(RAG_SETTINGS) != 2:
        raise ValueError("generate_table13 assumes exactly 2 RAG settings (baseline vs examples_solution).")
    setting_a, setting_b = RAG_SETTINGS

    # Families present in the given llms list, in order of first appearance
    family_order = list(dict.fromkeys(
        f for f, _ in (split_family_and_setting(l) for l in llms) if f
    ))

    # ------------------------------------------------------------------
    # 1. Run the paired Wilcoxon test per family, per quality dimension
    # ------------------------------------------------------------------
    raw_results = {metric: {} for metric in METRIC_FILES}  # {metric: {family: (p_value, median_a, median_b, n)}}

    for metric, filename in METRIC_FILES.items():
        path = f"{group_dir}/{filename}"
        df = pd.read_csv(path, sep=";", index_col=0)
        df.replace('', np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')

        for family in family_order:
            col_a = f"{family}_{setting_a}"
            col_b = f"{family}_{setting_b}"
            if col_a not in df.columns or col_b not in df.columns:
                print(f"[WARNING] Skipping {family} for {metric}: missing '{setting_a}' or '{setting_b}' column.")
                continue

            paired = df[[col_a, col_b]].dropna()
            n = len(paired)
            if n < MIN_PAIRED_CASES:
                print(f"[INFO] Skipping {family} for {metric}: only {n} paired cases (min {MIN_PAIRED_CASES}).")
                continue
            if (paired[col_a] == paired[col_b]).all():
                print(f"[INFO] Skipping {family} for {metric}: no differences between settings.")
                continue

            _, p_value = wilcoxon(paired[col_a], paired[col_b], zero_method='zsplit')
            raw_results[metric][family] = {
                "p_value": p_value,
                "median_a": paired[col_a].median(),
                "median_b": paired[col_b].median(),
                "n": n,
            }

    # ------------------------------------------------------------------
    # 2. Bonferroni-correct each metric column separately, across the
    #    family contrasts actually tested for that metric
    # ------------------------------------------------------------------
    adj_results = {metric: {} for metric in METRIC_FILES}
    for metric, per_family in raw_results.items():
        n_tests = len(per_family)
        for family, res in per_family.items():
            p_adj = min(res["p_value"] * n_tests, 1.0)
            adj_results[metric][family] = {**res, "p_adj": p_adj}

    # ------------------------------------------------------------------
    # 3. Assemble the table: one row per family
    # ------------------------------------------------------------------
    rows = []
    for family in family_order:
        row = {"RAG Contrast": f"{family}: {setting_a} → {setting_b}"}
        direction_parts = []

        for metric_col, label in zip(
            ["p_syntactic", "p_pragmatic", "p_semantic"], ["Syntactic", "Pragmatic", "Semantic"]
        ):
            res = adj_results[metric_col].get(family)
            if res is None:
                row[metric_col] = "n.s."
                direction_parts.append("–")
                continue

            row[metric_col] = format_p(res["p_value"], res["p_adj"])
            if res["p_adj"] < SIGNIFICANCE_ALPHA:
                winner = setting_b if res["median_b"] >= res["median_a"] else setting_a
                direction_parts.append(f"↑{winner}")
            else:
                direction_parts.append("–")

        row["Direction (Syntactic / Pragmatic / Semantic)"] = " / ".join(direction_parts)
        rows.append(row)

    table13_df = pd.DataFrame(rows)

    print("\n### Table 13: Wilcoxon signed-rank tests contrasting RAG settings (baseline vs examples_solution) ###")
    print("=" * 70)
    print(table13_df.to_string(index=False))

    outdir = f"{group_dir}/tables"
    save_table(table13_df, "table13", outdir)

    return table13_df