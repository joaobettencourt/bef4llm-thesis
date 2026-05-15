from .loaders import load_documents
from .chunking import chunk_documents


def get_rag_context(query, rag_config):
    print("\n[DEBUG] get_rag_context() called")
    print(f"[DEBUG] Query length: {len(query)}")

    rag_dir = rag_config.get("dir")
    top_k = rag_config.get("top_k", 3)

    print(f"[DEBUG] Documents directory: {rag_dir}")
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

        print(
            f"[DEBUG] Adding chunk {index} "
            f"(source={source}, size={len(chunk.text)})"
        )

        context += f"[SOURCE: {source}]\n"
        context += chunk.text + "\n\n"

    print(f"[DEBUG] Final context length: {len(context)}")

    return context