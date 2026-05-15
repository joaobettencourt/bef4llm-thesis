from .models import Chunk


def chunk_documents(documents, rag_config):
    print("\n[DEBUG] chunk_documents() called")

    chunk_size = rag_config.get("chunk_size", 500)
    overlap = rag_config.get("chunk_overlap", 100)

    print(f"[DEBUG] Chunk size: {chunk_size}")
    print(f"[DEBUG] Chunk overlap: {overlap}")

    chunks = []

    for document_index, document in enumerate(documents):

        source = document.metadata.get("source", "unknown")
        text = document.text

        print(
            f"[DEBUG] Document {document_index} "
            f"(source={source}) "
            f"text length: {len(text)}"
        )

        if not text.strip():
            print(f"[WARNING] Empty document text for source={source}")
            continue

        chunk_count_before = len(chunks)

        start = 0
        text_length = len(text)

        while start < text_length:

            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=document.metadata
                )
            )

            # move with overlap
            start += chunk_size - overlap

            if chunk_size <= overlap:
                raise ValueError(
                    "chunk_size must be greater than chunk_overlap"
                )

        created = len(chunks) - chunk_count_before

        print(
            f"[DEBUG] Created {created} chunks for source={source}"
        )

    print(f"[DEBUG] Total chunks: {len(chunks)}")

    return chunks