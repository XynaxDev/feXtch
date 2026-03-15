# =============================================================================
# feXtch — Fast EXtended file feTCH engine
# syncer.py  |  On-demand incremental index sync
#
# Author  : Akash Kumar
# Email   : akashkumar.cs27@gmail.com
# GitHub  : github.com/akashkumar
# Project : feXtch — local AI semantic file search
#
# Run this whenever you want the index to pick up new / modified / deleted files.
# It only re-embeds what actually changed — skips everything already indexed.
# =============================================================================

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_chroma import Chroma                  # updated — no deprecation warning
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    EMBED_BATCH_SIZE,
    EMBED_PARALLEL_BATCHES,
)
from scanner import scan_drives, get_scan_roots, _make_document


# ---------------------------------------------------------------------------
# Chroma connection
# ---------------------------------------------------------------------------

def _open_db() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        collection_name=CHROMA_COLLECTION,
    )


# ---------------------------------------------------------------------------
# Read existing index state
# ---------------------------------------------------------------------------

def _get_indexed_path_mtimes(db: Chroma) -> dict[str, str]:
    """Return {path: modified_at} for all docs currently in Chroma."""
    try:
        result = db.get(include=["metadatas"])
        return {
            m["path"]: m.get("modified_at", "")
            for m in result["metadatas"]
            if m and "path" in m
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public sync API
# ---------------------------------------------------------------------------

def sync(
    roots: list[str] | None = None,
    remove_deleted: bool = True,
) -> dict:
    """
    Incremental sync — only processes what changed since last run.

      new files     → embed and add
      modified files → delete stale entry, re-embed
      deleted files  → remove from index

    Args:
        roots          — paths to scan (None = auto from config)
        remove_deleted — whether to purge deleted file entries

    Returns:
        {"added": int, "updated": int, "removed": int, "elapsed": float}
    """
    t0    = time.perf_counter()
    roots = roots or get_scan_roots()
    db    = _open_db()

    print("[syncer] reading existing index …")
    indexed = _get_indexed_path_mtimes(db)
    print(f"[syncer] {len(indexed):,} paths currently indexed")

    print("[syncer] scanning filesystem …")
    docs, _, _ = scan_drives(roots=roots)

    new_docs:     list[Document] = []
    updated_docs: list[Document] = []

    for doc in docs:
        path        = doc.metadata["path"]
        modified_at = doc.metadata.get("modified_at", "")

        if path not in indexed:
            new_docs.append(doc)
        elif modified_at != indexed[path]:
            updated_docs.append(doc)

    disk_paths    = {doc.metadata["path"] for doc in docs}
    deleted_paths = set(indexed.keys()) - disk_paths

    print(
        f"[syncer] {len(new_docs):,} new  |  "
        f"{len(updated_docs):,} modified  |  "
        f"{len(deleted_paths):,} deleted"
    )

    to_embed = new_docs + updated_docs
    added = updated = removed = 0

    # ── Re-embed changed files ───────────────────────────────────────────
    if to_embed:
        # remove stale versions of modified files first
        for doc in updated_docs:
            try:
                db.delete(where={"path": doc.metadata["path"]})
            except Exception:
                pass

        n       = len(to_embed)
        batches = [to_embed[i: i + EMBED_BATCH_SIZE] for i in range(0, n, EMBED_BATCH_SIZE)]
        print(f"[syncer] embedding {n:,} docs …")

        def _push(batch, label):
            try:
                db.add_documents(batch)
                return label, True
            except Exception as e:
                print(f"[syncer] batch {label} failed: {e}")
                return label, False

        with ThreadPoolExecutor(max_workers=EMBED_PARALLEL_BATCHES) as pool:
            futs = {
                pool.submit(_push, b, f"{i+1}/{len(batches)}"): i
                for i, b in enumerate(batches)
            }
            for fut in as_completed(futs):
                label, ok = fut.result()
                if ok:
                    idx   = futs[fut]
                    start = idx * EMBED_BATCH_SIZE
                    end   = min(start + EMBED_BATCH_SIZE, n)
                    print(f"  [✓] batch {label}  ({start}–{end})")

        added   = len(new_docs)
        updated = len(updated_docs)

    # ── Remove deleted files ─────────────────────────────────────────────
    if remove_deleted and deleted_paths:
        print(f"[syncer] removing {len(deleted_paths):,} deleted entries …")
        for path in deleted_paths:
            try:
                db.delete(where={"path": path})
                removed += 1
            except Exception as e:
                print(f"[syncer] could not remove {path}: {e}")

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\n[syncer] done in {elapsed}s — +{added} new  ~{updated} updated  -{removed} removed")
    return {"added": added, "updated": updated, "removed": removed, "elapsed": elapsed}


if __name__ == "__main__":
    result = sync()
    print("sync summary:", result)