#!/usr/bin/env python
"""
build_vector_index.py

(Re)builds the semantic vector index used by RAG mode "examples", so
that generate_bpmn never pays the sentence-embedding cost at benchmark
runtime. Run this once, and again whenever rag/examples/ changes.

Usage (from repo root):
    python scripts/build_vector_index.py
    python scripts/build_vector_index.py --dir examples --force
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable even if the package isn't pip-installed,
# mirroring how the .sh scripts resolve paths relative to themselves.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.rag import vector_db

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the RAG examples vector index")
    parser.add_argument(
        "--dir",
        type=str,
        default="examples",
        help="Subfolder under Folder.RAG containing the text/bpmn pairs (default: 'examples')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if a valid cache already exists",
    )
    args = parser.parse_args()

    corpus_dir = Path(get_folder_path(Folder.RAG)) / args.dir
    print(f"[INFO] Building vector index for: {corpus_dir}")
    embeddings, items = vector_db.build_index(corpus_dir, force=args.force)
    print(f"[INFO] Done. Indexed {len(items)} pairs.")