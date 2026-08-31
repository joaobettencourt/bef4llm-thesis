import os
import pandas as pd


def format_pvalue(p):
    if p <= 0:
        return "< 1e-300"
    return f"{p:.2e}"


def format_conclusion(quality, significant):
    if not significant:
        return "Fail to reject H0: no significant difference"
    verb = "LLms" if quality == "Semantic" else "LLMs"
    return f"Reject H0: {verb} differ significantly"


def save_table(df, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    csv_path = f"{outdir}/{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")
    return csv_path