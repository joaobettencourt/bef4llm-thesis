from .loaders import load_documents, load_text
from .chunking import chunk_documents
from pathlib import Path
from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path

def get_rag_context(query, rag_config):
    print("\n[DEBUG] get_rag_context() called")
    print(f"[DEBUG] Query length: {len(query)}")

    rag_dir = rag_config.get("dir")
    rag_mode = rag_config.get("mode", "documents")
    top_k = rag_config.get("top_k", 3)

    print(f"[DEBUG] RAG mode: {rag_mode}")
    print(f"[DEBUG] Documents directory: {rag_dir}")

    directory = Path(get_folder_path(Folder.RAG)) / rag_dir

    if rag_mode == "mock":
        txt_files = list(directory.glob("*.txt"))
        if not txt_files:
            print("[WARNING] No .txt files found for mock mode")
            return ""
        context = ""
        for path in txt_files:
            doc = load_text(path)
            print(f"[DEBUG] Mock: loaded {path.name}")
            context += doc.text + "\n\n"
        return context

    # documents / examples modes (to be implemented)
    print(f"[DEBUG] Top-K: {top_k}")
    documents = load_documents(rag_dir)
    print(f"[DEBUG] Loaded documents: {len(documents)}")
    if not documents:
        print("[WARNING] No documents loaded")
        return ""
    chunks = chunk_documents(documents, rag_config)
    print(f"[DEBUG] Total chunks created: {len(chunks)}")
    if not chunks:
        print("[WARNING] No chunks created")
        return ""
    selected_chunks = chunks[:top_k]
    print(f"[DEBUG] Selected chunks: {len(selected_chunks)}")
    context = ""
    for index, chunk in enumerate(selected_chunks):
        source = chunk.metadata.get("source", "unknown")
        print(f"[DEBUG] Adding chunk {index} (source={source}, size={len(chunk.text)})")
        context += f"[SOURCE: {source}]\n"
        context += chunk.text + "\n\n"
    print(f"[DEBUG] Final context length: {len(context)}")
    return context