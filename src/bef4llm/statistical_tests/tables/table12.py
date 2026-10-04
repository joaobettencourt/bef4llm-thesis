import pandas as pd

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.statistical_tests.tables.common import save_table

# Order in which RAG settings should appear within each family group.
RAG_SETTINGS = ["baseline", "examples_rag", "examples_solution"]

# The three quality dimensions, and the CSV files they come from
# (same files already used by generate_table8).
METRIC_FILES = {
    "Qsyn": "syntactic quality.csv",
    "Qprag": "pragmatic quality.csv",
    "Qsem": "semantic quality.csv",
}


def split_family_and_setting(llm_column):
    """
    Splits a column name like 'eurollm-22b:latest_baseline' into
    ('eurollm-22b:latest', 'baseline'). Returns (None, None) if the
    column doesn't end with a known RAG setting suffix.
    """
    for setting in RAG_SETTINGS:
        suffix = f"_{setting}"
        if llm_column.endswith(suffix):
            family = llm_column[: -len(suffix)]
            return family, setting
    return None, None


def generate_table12(llms, runs):
    """
    Builds Table 12 (descriptive): mean Qsyn, Qprag, Qsem per LLM family,
    grouped by RAG setting (baseline vs examples_solution) instead of the
    paper's parameter-size variation. Within each family, the higher value
    per dimension is marked with an asterisk (*).

    Saved as table12.csv in statistical_tests_group_<runs>/tables.
    """
    runs_suffix = "_".join(str(r) for r in runs)
    group_dir = f"{get_folder_path(Folder.DATA)}/statistical_tests/statistical_tests_group_{runs_suffix}"

    # ------------------------------------------------------------------
    # 1. Column means per LLM config, for each quality dimension
    #    (same approach as generate_table8's quality_means block)
    # ------------------------------------------------------------------
    quality_means = {}  # {llm_column: {"Qsyn": ..., "Qprag": ..., "Qsem": ...}}
    for col_name, filename in METRIC_FILES.items():
        path = f"{group_dir}/{filename}"
        df = pd.read_csv(path, sep=";", index_col=0)
        means = df.mean(axis=0, skipna=True)
        for llm, value in means.items():
            quality_means.setdefault(llm, {})[col_name] = value

    # ------------------------------------------------------------------
    # 2. Group by family / RAG setting
    # ------------------------------------------------------------------
    rows = []
    for llm in llms:
        family, setting = split_family_and_setting(llm)
        if family is None:
            print(f"[WARNING] Skipping {llm}: doesn't match any known RAG setting suffix "
                  f"({RAG_SETTINGS}).")
            continue

        q = quality_means.get(llm)
        if q is None or any(pd.isna(q.get(m)) for m in METRIC_FILES):
            print(f"[WARNING] Skipping {llm}: missing quality data.")
            continue

        rows.append({
            "Family": family,
            "RAG setting": setting,
            "Qsyn": round(q["Qsyn"], 3),
            "Qprag": round(q["Qprag"], 3),
            "Qsem": round(q["Qsem"], 3),
        })

    table12_df = pd.DataFrame(rows)

    if table12_df.empty:
        print("No families/settings had complete data for Table 12.")
        save_table(table12_df, "table12", f"{group_dir}/tables")
        return table12_df

    # ------------------------------------------------------------------
    # 3. Mark the higher value per dimension within each family with "*"
    # ------------------------------------------------------------------
    def mark_higher(group):
        group = group.copy()
        for metric in ["Qsyn", "Qprag", "Qsem"]:
            values = group[metric]
            if values.notna().sum() < 2:
                formatted = [f"{v:.3f}" for v in values]
            else:
                max_idx = values.idxmax()
                formatted = [
                    f"{v:.3f}*" if idx == max_idx else f"{v:.3f}"
                    for idx, v in values.items()
                ]
            group[metric] = formatted
        return group

    # Preserve family order as given in `llms`, and baseline/examples_solution order within each
    family_order = list(dict.fromkeys(f for f, _ in (split_family_and_setting(l) for l in llms) if f))
    table12_df["Family"] = pd.Categorical(table12_df["Family"], categories=family_order, ordered=True)
    table12_df["RAG setting"] = pd.Categorical(table12_df["RAG setting"], categories=RAG_SETTINGS, ordered=True)
    table12_df = table12_df.sort_values(["Family", "RAG setting"]).reset_index(drop=True)

    table12_df = table12_df.groupby("Family", group_keys=False, observed=True).apply(mark_higher)

    print("\n### Table 12: Mean syntactic, pragmatic, and semantic scores per RAG setting ###")
    print("=" * 70)
    print(table12_df.to_string(index=False))

    outdir = f"{group_dir}/tables"
    save_table(table12_df, "table12", outdir)

    return table12_df