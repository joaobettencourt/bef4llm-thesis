"""
Semantic vector index over (description, BPMN) example pairs.

Given a corpus directory containing one or more subfolders of paired
`<name>.txt` / `<name>.bpmn` files, this module builds (and persists)
sentence embeddings for every description, and retrieves the example
whose description is semantically closest to a given query.

The index is cached to disk under `<corpus_dir>/.vector_index/` so
embeddings are not recomputed on every run. The cache is automatically
rebuilt if the set of files (or their modification times) changes, but
in normal operation it should always be pre-built via
`scripts/build_vector_index.py` so that benchmark runs never pay
the embedding cost at runtime.

Two layers protect against RAG leakage (handing the LLM the ground
truth disguised as a "similar example"):
  1. text_hash: exact-duplicate detection (after whitespace/case
     normalization) between the query and any corpus item.
  2. duplicate_groups.json: a manually curated list of near-duplicate
     example names across the corpus (e.g. the same exercise appearing
     in two datasets under different filenames, with minor wording
     differences that text_hash won't catch). Generate candidates with
     scripts/detect_duplicate_examples.py, then review by hand.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import re

_MODEL_NAME = "all-MiniLM-L6-v2"
_INDEX_DIRNAME = ".vector_index"
_DUPLICATE_GROUPS_FILENAME = "duplicate_groups.json"
_SCHEMA_VERSION = 2  # bump whenever the cached item structure changes

_model = None            # lazy singleton for the embedding model
_memory_cache = {}       # {corpus_dir_str: (embeddings, items, fingerprint)}



_BPMN_DI_BLOCK_RE = re.compile(
    r"\s*<bpmndi:BPMNDiagram\b.*?</bpmndi:BPMNDiagram>",
    re.DOTALL,
)


def _strip_diagram_interchange(bpmn_xml: str) -> str:
    """
    Removes the <bpmndi:BPMNDiagram>...</bpmndi:BPMNDiagram> block (pure
    visual layout info — coordinates, bounds, waypoints) from a BPMN 2.0 XML
    string, keeping only the semantic process/collaboration definition.
    Every corpus file uses the standard 'bpmndi' prefix, so a direct text
    match is used instead of a full XML parse/re-serialize — this keeps the
    rest of the file byte-for-byte identical to the source.
    """
    return _BPMN_DI_BLOCK_RE.sub("", bpmn_xml)


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[DEBUG] vector_db: loading embedding model '{_MODEL_NAME}'")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _normalize_text(text: str) -> str:
    """Collapse whitespace and lowercase, so trivial formatting differences
    (extra spaces, line breaks, casing) don't defeat duplicate detection."""
    return " ".join(text.split()).strip().lower()


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _load_duplicate_groups(corpus_dir: Path):
    """
    Loads corpus_dir/duplicate_groups.json (a manually curated list of
    equivalent example names, e.g. [["02-Recourse", "Recourse"], ...])
    and returns {pair_name: set(other names in its group)}.
    Returns {} if the file doesn't exist or fails to parse.
    """
    path = corpus_dir / _DUPLICATE_GROUPS_FILENAME
    if not path.exists():
        return {}
    try:
        groups = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARNING] vector_db: failed to load {path} ({e}), ignoring duplicate groups")
        return {}

    lookup = {}
    for group in groups:
        group_set = set(group)
        for name in group:
            lookup[name] = group_set - {name}
    return lookup


def _collect_pairs(corpus_dir: Path):
    """Recursively finds every {name}.txt with a matching {name}.bpmn."""
    pairs = []
    for txt_path in sorted(corpus_dir.rglob("*.txt")):
        bpmn_path = txt_path.with_suffix(".bpmn")
        if not bpmn_path.exists():
            print(f"[WARNING] vector_db: no matching .bpmn for {txt_path}")
            continue
        pairs.append((txt_path.stem, txt_path, bpmn_path))
    return pairs


def _fingerprint(pairs):
    """Cheap signature (paths + mtimes) to detect a stale cache."""
    parts = []
    for _, txt_path, bpmn_path in pairs:
        parts.append(f"{txt_path}:{txt_path.stat().st_mtime_ns}")
        parts.append(f"{bpmn_path}:{bpmn_path.stat().st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _index_paths(corpus_dir: Path):
    index_dir = corpus_dir / _INDEX_DIRNAME
    return index_dir, index_dir / "embeddings.npy", index_dir / "metadata.json"


def _load_cached_index(corpus_dir: Path):
    _, emb_path, meta_path = _index_paths(corpus_dir)
    if not (emb_path.exists() and meta_path.exists()):
        return None
    try:
        embeddings = np.load(emb_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return embeddings, metadata
    except Exception as e:
        print(f"[WARNING] vector_db: failed to load cached index ({e}), rebuilding")
        return None


def _save_index(corpus_dir: Path, embeddings, metadata):
    index_dir, emb_path, meta_path = _index_paths(corpus_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_index(corpus_dir: Path):
    pairs = _collect_pairs(corpus_dir)
    if not pairs:
        print(f"[WARNING] vector_db: no text/bpmn pairs found under {corpus_dir}")
        return np.zeros((0, 0)), []

    print(f"[DEBUG] vector_db: building index for {len(pairs)} pairs in {corpus_dir}")
    model = _get_model()

    texts = [txt_path.read_text(encoding="utf-8") for _, txt_path, _ in pairs]
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    )

    items = [
        {
            "pair": stem,
            "txt_path": str(txt_path),
            "bpmn_path": str(bpmn_path),
            "text_hash": _text_hash(text),
        }
        for (stem, txt_path, bpmn_path), text in zip(pairs, texts)
    ]
    metadata = {
        "fingerprint": _fingerprint(pairs),
        "model": _MODEL_NAME,
        "schema_version": _SCHEMA_VERSION,
        "items": items,
    }

    _save_index(corpus_dir, embeddings, metadata)
    return embeddings, items


def _get_index(corpus_dir: Path):
    """Returns (embeddings, items), rebuilding on-disk/in-memory cache if stale."""
    key = str(corpus_dir.resolve())
    pairs = _collect_pairs(corpus_dir)
    fingerprint = _fingerprint(pairs)

    cached_mem = _memory_cache.get(key)
    if cached_mem is not None and cached_mem[2] == fingerprint:
        return cached_mem[0], cached_mem[1]

    cached_disk = _load_cached_index(corpus_dir)
    if cached_disk is not None:
        embeddings, meta = cached_disk
        if (
            meta.get("fingerprint") == fingerprint
            and meta.get("model") == _MODEL_NAME
            and meta.get("schema_version") == _SCHEMA_VERSION
        ):
            print(f"[DEBUG] vector_db: using cached index ({len(meta['items'])} items)")
            _memory_cache[key] = (embeddings, meta["items"], fingerprint)
            return embeddings, meta["items"]
        print("[WARNING] vector_db: cache is stale (corpus changed or schema updated)")

    print("[WARNING] vector_db: no valid pre-built index found, building now "
          "(consider running scripts/build_vector_index.py ahead of time)")
    embeddings, items = _compute_index(corpus_dir)
    _memory_cache[key] = (embeddings, items, fingerprint)
    return embeddings, items


def build_index(corpus_dir: Path, force: bool = False):
    """
    Explicitly builds (or refreshes) the on-disk index for corpus_dir.
    Intended to be called from scripts/build_vector_index.py so that
    get_most_similar_semantically() always finds a fresh, valid cache at
    benchmark runtime and never pays the embedding cost mid-run.
    """
    corpus_dir = Path(corpus_dir)
    if force:
        print("[INFO] vector_db: --force set, rebuilding index unconditionally")
        embeddings, items = _compute_index(corpus_dir)
        key = str(corpus_dir.resolve())
        pairs = _collect_pairs(corpus_dir)
        _memory_cache[key] = (embeddings, items, _fingerprint(pairs))
        return embeddings, items
    return _get_index(corpus_dir)


def get_most_similar_semantically(query: str, corpus_dir: Path, exclude_pair: str = None,
                                    return_scores: bool = False, exclude_exact_duplicates: bool = True):
    """
    ...
    If return_scores=True, returns (formatted_string, candidates) where
    candidates lists ALL non-excluded corpus items with their cosine
    similarity score, sorted descending (candidates[0] is the best match,
    used to build formatted_string).
    """
    corpus_dir = Path(corpus_dir)
    embeddings, items = _get_index(corpus_dir)
    empty = ("", []) if return_scores else ""
    if len(items) == 0:
        return empty

    duplicate_groups = _load_duplicate_groups(corpus_dir)
    exclude_names = set()
    if exclude_pair is not None:
        exclude_names.add(exclude_pair)
        exclude_names |= duplicate_groups.get(exclude_pair, set())

    query_text_hash = _text_hash(query)

    model = _get_model()
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )[0]

    scores = embeddings @ query_embedding
    order = np.argsort(-scores)

    candidates = []
    best_item = None

    for idx in order:
        item = items[idx]
        if item["pair"] in exclude_names:
            continue
        if exclude_exact_duplicates and item.get("text_hash") == query_text_hash:
            continue
        if best_item is None:
            best_item = item
            print(f"[DEBUG] vector_db: best match = {item['pair']} (score={scores[idx]:.4f})")
        candidates.append({"pair": item["pair"], "score": float(scores[idx])})
        # no break — keep going through every remaining item

    if best_item is None:
        print("[WARNING] vector_db: no candidate left after exclusions")
        return empty

    txt_content = Path(best_item["txt_path"]).read_text(encoding="utf-8")
    bpmn_content = _strip_diagram_interchange(Path(best_item["bpmn_path"]).read_text(encoding="utf-8"))
    result = f"[DESCRIPTION:]\n{txt_content}\n\n[BPMN MODEL:]\n{bpmn_content}\n\n"

    if return_scores:
        return result, candidates
    return result