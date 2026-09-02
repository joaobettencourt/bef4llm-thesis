#!/usr/bin/env python
"""
detect_duplicate_examples.py

Scans the RAG examples corpus for near-duplicate descriptions across
datasets (the same underlying process described in more than one file,
e.g. "02-Recourse" in camunda/ and "Recourse" in lre_old/). A duplicate
description leaks the ground-truth BPMN into the RAG context whenever
it gets picked as the "most similar" example for a different-named
pair describing the same process — exclude_pair alone (name match)
doesn't catch this.

This script does NOT modify anything by default. It prints candidate
duplicate pairs (cosine similarity above --threshold) for manual
review — open the .txt files yourself before trusting a match, since a
high score can also mean "very similar process, different case", not
a true duplicate. Use --write to also dump a starter JSON you then
edit/confirm by hand.

Usage (from repo root):
    python scripts/detect_duplicate_examples.py
    python scripts/detect_duplicate_examples.py --threshold 0.85
    python scripts/detect_duplicate_examples.py --write rag/examples/duplicate_groups.json
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.rag import vector_db


def _dataset_of(item):
    """Dataset name = parent folder of the .txt file (e.g. 'camunda', 'lre_old')."""
    return Path(item["txt_path"]).parent.name


def main():
    parser = argparse.ArgumentParser(description="Detect near-duplicate RAG examples")
    parser.add_argument("--dir", type=str, default="examples",
                         help="Subfolder under Folder.RAG to scan (default: 'examples')")
    parser.add_argument("--threshold", type=float, default=0.90,
                         help="Cosine similarity above which a pair is flagged (default: 0.90)")
    parser.add_argument("--write", type=str, default=None,
                         help="If set, writes a starter duplicate_groups.json to this path")
    args = parser.parse_args()

    corpus_dir = Path(get_folder_path(Folder.RAG)) / args.dir
    embeddings, items = vector_db.build_index(corpus_dir)  # reuses cache if valid
    n = len(items)
    print(f"[INFO] Comparing {n} examples pairwise (threshold={args.threshold})")

    sim_matrix = embeddings @ embeddings.T
    flagged = []
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j]
            if score >= args.threshold:
                flagged.append((float(score), items[i], items[j]))

    flagged.sort(key=lambda x: -x[0])

    if not flagged:
        print("[INFO] No candidate duplicates found above threshold.")
        return

    print(f"\n[RESULT] {len(flagged)} candidate duplicate pair(s):\n")
    groups = []
    for score, a, b in flagged:
        print(f"  {score:.4f}  {a['pair']:<25} ({_dataset_of(a):<15}) <-> "
              f"{b['pair']:<25} ({_dataset_of(b)})")
        groups.append([a["pair"], b["pair"]])

    print(
        "\n[NOTE] Review each pair above manually (diff the .txt files) before trusting it. "
        "Merge/remove entries in the output JSON as needed — this script only proposes candidates."
    )

    if args.write:
        out_path = Path(args.write)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[INFO] Wrote starter duplicate groups to {out_path} "
              f"— edit it (merge groups, drop false positives) before relying on it.")


if __name__ == "__main__":
    main()
