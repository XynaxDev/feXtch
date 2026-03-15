# =============================================================================
# feXtch — Fast EXtended file feTCH engine
# indexer.py  |  Two-phase embedding pipeline + Chroma persistence
#
# Author  : Akash Kumar
# Email   : akashkumar.cs27@gmail.com
# GitHub  : github.com/akashkumar
# Project : feXtch — local AI semantic file search
# =============================================================================

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    EMBED_BATCH_SIZE,
    EMBED_PARALLEL_BATCHES,
    PHASE1_QUICK_INDEX,
)
from scanner import scan_drives


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _push_batch(batch: list[Document], db: Chroma, label: str) -> tuple[str, bool]:
    try:
        db.add_documents(batch)
        return label, True
    except Exception as err:
        print(f"[indexer] batch {label} failed: {err}")
        return label, False


def _embed_phase(
    docs: list[Document],
    db: Chroma,
    phase_name: str,
    start_offset: int = 0,
) -> int:
    n = len(docs)
    if n == 0:
        return 0

    batches       = list(_chunks(docs, EMBED_BATCH_SIZE))
    total_batches = len(batches)
    succeeded     = 0

    with ThreadPoolExecutor(max_workers=EMBED_PARALLEL_BATCHES) as pool:
        future_to_idx = {
            pool.submit(_push_batch, batch, db, f"{i+1}/{total_batches}"): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(future_to_idx):
            label, ok = future.result()
            idx   = future_to_idx[future]
            start = start_offset + idx * EMBED_BATCH_SIZE
            end   = min(start + EMBED_BATCH_SIZE, start_offset + n)
            status = "✓" if ok else "✗"
            print(f"  [{status}] {phase_name} batch {label}  (docs {start}–{end})")
            if ok:
                succeeded += 1

    return min(succeeded * EMBED_BATCH_SIZE, n)


def _background_phase(docs: list[Document], db: Chroma, phase1_count: int):
    print(f"\n[indexer] ⟳ background: embedding {len(docs):,} remaining docs …")
    count = _embed_phase(docs, db, "bg", start_offset=phase1_count)
    print(f"[indexer] ✓ background done — {count:,} additional docs persisted")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(
    roots: list[str] | None = None,
    phase1_count: int = PHASE1_QUICK_INDEX,
) -> tuple[dict[str, int], int]:
    """
    Scan → score → embed in two phases.

    IMPORTANT: always deletes and recreates the Chroma collection before
    indexing.  This prevents duplicate documents that accumulate when the
    indexer is run multiple times on the same collection — duplicates caused
    the same file to appear 8+ times in every query result.
    """
    t0 = time.perf_counter()

    docs, stats, _ = scan_drives(roots=roots)

    if not docs:
        print("[indexer] nothing to embed — check scan roots or filters.")
        return stats, 0

    n = len(docs)
    print(f"\n[indexer] {n:,} documents to embed")
    print(f"[indexer] phase 1: indexing top {min(phase1_count, n):,} files now …")
    print(f"[indexer] phase 2: {max(0, n - phase1_count):,} files in background\n")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Delete the collection first so re-running never produces duplicates.
    # Chroma auto-creates it again on the next add_documents call.
    try:
        client_db = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION,
        )
        client_db.delete_collection()
        print("[indexer] cleared existing collection\n")
    except Exception:
        pass

    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )

    # Phase 1 — synchronous
    phase1_docs    = docs[:phase1_count]
    phase1_indexed = _embed_phase(phase1_docs, db, "p1")

    elapsed = time.perf_counter() - t0
    print(f"\n[indexer] ✓ phase 1 done in {elapsed:.1f}s")
    print(f"[indexer] {phase1_indexed:,} docs ready — you can start searching now")
    print(f"[indexer] index at: {CHROMA_DIR.resolve()}\n")

    # Phase 2 — background
    phase2_docs = docs[phase1_count:]
    if phase2_docs:
        t = threading.Thread(
            target=_background_phase,
            args=(phase2_docs, db, len(phase1_docs)),
            daemon=True,
        )
        t.start()
        print(f"[indexer] ⟳ background thread started for {len(phase2_docs):,} remaining docs")

    return stats, phase1_indexed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stats, ready = build_index()
    print("\ndrive stats:", stats)
    print("immediately searchable:", ready)

    print("\nPress Ctrl+C to stop (background indexing will stop too)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nbye.")