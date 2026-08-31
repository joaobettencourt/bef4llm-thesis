import pandas as pd
import os

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path

from bef4llm.statistical_tests.syntactic_quality_analysis import run_syntactic_statistical_tests
from bef4llm.statistical_tests.pragmatic_quality_analysis import run_pragmatic_statistical_tests
from bef4llm.statistical_tests.semantic_quality_analysis import run_semantic_statistical_tests
from bef4llm.statistical_tests.tables.common import format_pvalue, format_conclusion, save_table


def generate_table7(runs):
    """
    Runs the Skillings-Mack global test for all three quality dimensions
    and saves the result as Table 7 (csv), in the same
    statistical_tests_group_<runs> directory used by the other outputs,
    under a 'tables' subfolder.
    """
    dimensions = [
        ("Syntactic", run_syntactic_statistical_tests),
        ("Pragmatic", run_pragmatic_statistical_tests),
        ("Semantic", run_semantic_statistical_tests),
    ]

    rows = []
    for name, func in dimensions:
        sm_stat, sm_p, sm_df = func(runs)
        rows.append({
            "Quality Dimension": name,
            "Test Statistic (T)": round(sm_stat, 2),
            "df": sm_df,
            "p-value": format_pvalue(sm_p),
            "Conclusion": format_conclusion(name, sm_p < 0.05),
        })

    table7_df = pd.DataFrame(rows)

    print("\n### Table 7: Global Skillings-Mack test results across quality dimensions ###")
    print("=" * 70)
    print(table7_df.to_string(index=False))

    runs_suffix = "_".join(str(r) for r in runs)
    outdir = f"{get_folder_path(Folder.DATA)}/statistical_tests/statistical_tests_group_{runs_suffix}/tables"

    save_table(table7_df, "table7", outdir)

    return table7_df