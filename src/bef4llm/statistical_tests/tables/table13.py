import itertools

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.statistical_tests.tables.common import save_table
from bef4llm.statistical_tests.tables.table12 import split_family_and_setting, RAG_SETTINGS

METRIC_FILES = {
    "p_syntactic": "syntactic quality.csv",
    "p_pragmatic": "pragmatic quality.csv",
    "p_semantic": "semantic quality.csv",
}

MIN_PAIRED_CASES = 10
SIGNIFICANCE_ALPHA = 0.05


def format_p(p, adj):
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
    Builds Table 13 (Wilcoxon signed-rank tests contrasting every pair of
    RAG settings within each LLM family, Bonferroni-adjusted separately per
    quality dimension across all family/pair contrasts tested for that
    dimension), analogous to the paper's Table 13 which contrasts small vs
    large checkpoints within each LLM family.

    With N RAG settings, each family gets C(N,2) rows (one per unordered
    pair, in RAG_SETTINGS order) instead of a single baseline-vs-treatment
    row.

    Saved as table13.csv in statistical_tests_group_<runs>/tables.
    """
    runs_suffix = "_".join(str(r) for r in runs)
    group_dir = f"{get_folder_path(Folder.DATA)}/statistical_tests/statistical_tests_group_{runs_suffix}"

    if len(RAG_SETTINGS) < 2:
        raise ValueError("generate_table13 needs at least 2 RAG settings to contrast.")
    setting_pairs = list(itertools.combinations(RAG_SETTINGS, 2))

    family_order = list(dict.fromkeys(
        f for f, _ in (split_family_and_setting(l) for l in llms) if f
    ))

    # ------------------------------------------------------------------
    # 1. Run the paired Wilcoxon test per family, per setting pair, per
    #    quality dimension
    # ------------------------------------------------------------------
    raw_results = {metric: {} for metric in METRIC_FILES}  # {metric: {(family, a, b): {...}}}

    for metric, filename in METRIC_FILES.items():
        path = f"{group_dir}/{filename}"
        df = pd.read_csv(path, sep=";", index_col=0)
        df.replace('', np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')

        for family in family_order:
            for setting_a, setting_b in setting_pairs:
                col_a = f"{family}_{setting_a}"
                col_b = f"{family}_{setting_b}"
                if col_a not in df.columns or col_b not in df.columns:
                    print(f"[WARNING] Skipping {family} ({setting_a} vs {setting_b}) for {metric}: "
                          f"missing '{setting_a}' or '{setting_b}' column.")
                    continue

                paired = df[[col_a, col_b]].dropna()
                n = len(paired)
                if n < MIN_PAIRED_CASES:
                    print(f"[INFO] Skipping {family} ({setting_a} vs {setting_b}) for {metric}: "
                          f"only {n} paired cases (min {MIN_PAIRED_CASES}).")
                    continue
                if (paired[col_a] == paired[col_b]).all():
                    print(f"[INFO] Skipping {family} ({setting_a} vs {setting_b}) for {metric}: "
                          f"no differences between settings.")
                    continue

                _, p_value = wilcoxon(paired[col_a], paired[col_b], zero_method='zsplit')
                raw_results[metric][(family, setting_a, setting_b)] = {
                    "p_value": p_value,
                    "median_a": paired[col_a].median(),
                    "median_b": paired[col_b].median(),
                    "n": n,
                }

    # ------------------------------------------------------------------
    # 2. Bonferroni-correct each metric column separately, across every
    #    (family, setting pair) contrast actually tested for that metric
    # ------------------------------------------------------------------
    adj_results = {metric: {} for metric in METRIC_FILES}
    for metric, per_contrast in raw_results.items():
        n_tests = len(per_contrast)
        for key, res in per_contrast.items():
            p_adj = min(res["p_value"] * n_tests, 1.0)
            adj_results[metric][key] = {**res, "p_adj": p_adj}

    # ------------------------------------------------------------------
    # 3. Assemble the table: one row per family per setting pair
    # ------------------------------------------------------------------
    rows = []
    for family in family_order:
        for setting_a, setting_b in setting_pairs:
            row = {"RAG Contrast": f"{family}: {setting_a} → {setting_b}"}
            direction_parts = []

            for metric_col in ["p_syntactic", "p_pragmatic", "p_semantic"]:
                res = adj_results[metric_col].get((family, setting_a, setting_b))
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

    print("\n### Table 13: Wilcoxon signed-rank tests contrasting RAG settings ###")
    print("=" * 70)
    print(table13_df.to_string(index=False))

    outdir = f"{group_dir}/tables"
    save_table(table13_df, "table13", outdir)

    return table13_df