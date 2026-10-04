from .loaders import load_text, load_documents
from .chunking import chunk_documents
from pathlib import Path
from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
import bef4llm.llm_comparison.promting_helper as prompts
from . import vector_db


def _get_mock_examples_context(pair, directory):
    """
    Loads the .bpmn and .txt files matching `pair` from the given directory.
    """
    bpmn_path = directory / f"{pair}.bpmn"
    txt_path = directory / f"{pair}.txt"

    if not bpmn_path.exists():
        print(f"[WARNING] No .bpmn file found for pair '{pair}' in {directory}")
        return ""

    context = ""

    if txt_path.exists():
        doc = load_text(txt_path)
        print(f"[DEBUG] mock_examples: loaded text {txt_path.name}")
        context += f"[DESCRIPTION:]\n{doc.text}\n\n"
    else:
        print(f"[WARNING] No .txt found alongside {bpmn_path.name}")

    bpmn_content = bpmn_path.read_text(encoding="utf-8")
    print(f"[DEBUG] mock_examples: loaded BPMN {bpmn_path.name}")
    context += f"[BPMN MODEL:]\n{bpmn_content}\n\n"

    print(f"[DEBUG] Final context length: {len(context)}")
    return context


def get_rag_context(query, rag_config, pair=None, dataset=None):
    print("\n[DEBUG] get_rag_context() called")
    print(f"[DEBUG] Query length: {len(query)}")

    rag_dir = rag_config.get("dir")
    rag_mode = rag_config.get("mode", "documents")
    top_k = rag_config.get("top_k", 3)

    print(f"[DEBUG] RAG mode: {rag_mode}")
    print(f"[DEBUG] Documents directory: {rag_dir}")

    base_dir = Path(get_folder_path(Folder.RAG)) / rag_dir

    if rag_mode == "examples":
        best_example, candidates = vector_db.get_most_similar_semantically(
            query, corpus_dir=base_dir, exclude_pair=pair, return_scores=True
        )
        if not best_example.strip():
            return "", None
        similarity_info = {
            "mode": "examples",
            "query_pair": pair,
            "best_match": candidates[0]["pair"] if candidates else None,
            "best_score": candidates[0]["score"] if candidates else None,
            "candidates": candidates,
        }
        return prompts.rag_context_connector_examples + best_example, similarity_info


    if rag_mode == "mock_examples":
        best_example, candidates = vector_db.get_most_similar_semantically(
            query, corpus_dir=base_dir, exclude_pair=None,
            exclude_exact_duplicates=False, return_scores=True
        )

        if not best_example.strip():
            return "", None
        similarity_info = {
            "mode": "mock_examples",
            "query_pair": pair,
            "best_match": candidates[0]["pair"] if candidates else None,
            "best_score": candidates[0]["score"] if candidates else None,
            "candidates": candidates,
        }
        return prompts.rag_context_connector_examples + best_example, similarity_info

    raise ValueError(f"Unknown RAG mode: '{rag_mode}'. Expected 'examples' or 'mock_examples'.")
