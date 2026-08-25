import pandas as pd
import numpy as np

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.statistical_tests.tables.common import save_table

from bef4llm.statistical_tests.semantic_quality_analysis import (
    run_pairwise_wilcoxon,
    calculate_descriptive_stats,
)

SIGNIFICANCE_ALPHA = 0.05


def generate_table11(runs):
    """
    Builds Table 11 (Wilcoxon contrasts for the semantic quality,
    Bonferroni-adjusted) using the SAME run_pairwise_wilcoxon and
    calculate_descriptive_stats logic already used for the console report,
    and saves it as table11.csv in statistical_tests_group_<runs>/tables.

    Format matches the paper (page 36):
        Better model | Worse model | padj | Paired cases
    followed by a summary row "All remaining N pairs: n.s."
    """
    runs_suffix = "_".join(str(r) for r in runs)
    group_dir = f"{get_folder_path(Folder.DATA)}/statistical_tests/statistical_tests_group_{runs_suffix}"

    path = f"{group_dir}/semantic quality.csv"
    data = pd.read_csv(path, sep=";", index_col=0)
    data.replace('', np.nan, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')

    # --- reuse existing logic, unchanged ---
    pairwise_results = run_pairwise_wilcoxon(data)          # all admissible pairs, Bonferroni-adjusted
    descriptive_df = calculate_descriptive_stats(data)      # median per model, for direction
    medians = descriptive_df.set_index("Model")["Median"]

    if pairwise_results.empty:
        print("No pairs had sufficient data for comparison.")
        empty_df = pd.DataFrame(columns=["Better model", "Worse model", "padj", "Paired cases"])
        save_table(empty_df, "table11", f"{group_dir}/tables")
        return empty_df

    total_pairs = len(pairwise_results)
    significant = pairwise_results[pairwise_results["p_value_corr"] < SIGNIFICANCE_ALPHA].copy()
    n_significant = len(significant)
    n_remaining = total_pairs - n_significant

    # --- only new part: derive Better/Worse from the medians already computed ---
    def better_worse(row):
        m1, m2 = row["Model 1"], row["Model 2"]
        return (m1, m2) if medians[m1] >= medians[m2] else (m2, m1)

    significant[["Better model", "Worse model"]] = significant.apply(
        lambda row: pd.Series(better_worse(row)), axis=1
    )

    table11_df = significant[["Better model", "Worse model", "p_value_corr", "n_pairs"]].rename(
        columns={"p_value_corr": "padj", "n_pairs": "Paired cases"}
    ).sort_values("padj").reset_index(drop=True)

    # Paired cases are counts -> plain integers, no decimals
    table11_df["Paired cases"] = table11_df["Paired cases"].astype(int)

    # padj -> scientific notation, 2 decimal places, "x × 10^y" style (as in the paper)
    def format_padj(p):
        mantissa, exponent = f"{p:.2e}".split("e")
        return f"{float(mantissa):.2f} × 10{int(exponent)}"

    table11_df["padj"] = table11_df["padj"].apply(format_padj)

    # summary row, exactly like Table 11 in the paper ("All remaining N pairs: n.s.")
    summary_row = pd.DataFrame([{
        "Better model": f"All remaining {n_remaining} pairs",
        "Worse model": "n.s.",
        "padj": "",
        "Paired cases": "",
    }])
    table11_df = pd.concat([table11_df, summary_row], ignore_index=True)

    print("\n### Table 11: Wilcoxon contrasts for the semantic quality (Bonferroni-adjusted) ###")
    print("=" * 70)
    print(f"{n_significant} out of {total_pairs} contrasts yield statistically significant results.")
    print(table11_df.to_string(index=False))

    outdir = f"{group_dir}/tables"
    save_table(table11_df, "table11", outdir)

    return table11_df