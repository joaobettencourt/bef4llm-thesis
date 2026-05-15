from pathlib import Path
from pypdf import PdfReader
from .models import Document

from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path


def load_pdf(path):

    print(f"[DEBUG] Loading PDF: {path.name}")

    reader = PdfReader(str(path))

    print(f"[DEBUG] PDF pages: {len(reader.pages)}")

    text = ""

    for page_index, page in enumerate(reader.pages):

        extracted = page.extract_text()

        extracted_length = len(extracted) if extracted else 0

        print(
            f"[DEBUG] Page {page_index} "
            f"extracted chars: {extracted_length}"
        )

        if extracted:
            text += extracted + "\n"

    print(f"[DEBUG] Total extracted chars: {len(text)}")

    return Document(
        text=text,
        metadata={
            "source": path.name,
            "type": "pdf"
        }
    )


def load_text(path):

    with open(path, "r", encoding="utf-8") as f:

        text = f.read()

    return Document(
        text=text,
        metadata={
            "source": path.name,
            "type": path.suffix.lower()[1:]
        }
    )


EXTRACTORS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
}


def load_documents(directory):

    documents = []

    directory = Path(get_folder_path(Folder.RAG)) / directory

    print(f"[DEBUG] Loading documents from: {directory}")

    for path in directory.iterdir():

        if not path.is_file():
            continue

        print(f"[DEBUG] Inspecting: {path.name}")
        print(f"[DEBUG] File suffix: '{path.suffix.lower()}'")

        extractor = EXTRACTORS.get(path.suffix.lower())

        if not extractor:

            print(f"[WARNING] No extractor for: {path.name}")

            continue

        try:

            document = extractor(path)

            documents.append(document)

            print(f"[DEBUG] Loaded: {path.name}")

        except Exception as e:

            print(f"[ERROR] Failed to load {path.name}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")

    return documents