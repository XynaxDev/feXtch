# =============================================================================
# feXtch — Fast EXtended file feTCH engine
# scanner.py  |  Cross-OS filesystem walker + document builder + scorer
#
# Author  : Akash Kumar
# Email   : akashkumar.cs27@gmail.com
# GitHub  : github.com/XynaxDev
# Project : feXtch — local AI semantic file search
#
# Virtual environment detection
# ──────────────────────────────
# Skipping venvs by directory name (venv, .venv, env) only catches
# conventionally-named environments.  Projects commonly use custom names:
#   myenv, myenv311, projenv, aqienv, voiceagent, flask_env, ...
#
# Every Python virtual environment — regardless of its name — contains
# a pyvenv.cfg file at its root.  This is created by the venv module
# and is the definitive marker that a directory is a venv.
#
# Fix: during os.walk, before descending into any subdirectory, check
# whether it contains pyvenv.cfg.  If it does, prune it from dirs[].
# This catches every custom-named venv without hardcoding any names.
# =============================================================================

import os
import platform
import string
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from config import (
    CUSTOM_SCAN_ROOTS,
    SKIP_DIRS_COMMON,
    SKIP_DIRS_WINDOWS,
    SKIP_DIRS_LINUX,
    SKIP_DIRS_MACOS,
    SKIP_PATH_FRAGMENTS_WINDOWS,
    SKIP_PATH_FRAGMENTS_LINUX,
    SKIP_PATH_FRAGMENTS_MACOS,
    SKIP_EXTENSIONS,
    SKIP_PREFIX,
    EXTENSION_PRIORITY,
    DEFAULT_EXTENSION_PRIORITY,
    SCORE_WEIGHTS,
)

_OS = platform.system()

if _OS == "Windows":
    _SKIP_DIRS_ALL = SKIP_DIRS_COMMON | SKIP_DIRS_WINDOWS
    _SKIP_FRAGS    = SKIP_PATH_FRAGMENTS_WINDOWS
elif _OS == "Darwin":
    _SKIP_DIRS_ALL = SKIP_DIRS_COMMON | SKIP_DIRS_MACOS
    _SKIP_FRAGS    = SKIP_PATH_FRAGMENTS_MACOS
else:
    _SKIP_DIRS_ALL = SKIP_DIRS_COMMON | SKIP_DIRS_LINUX
    _SKIP_FRAGS    = SKIP_PATH_FRAGMENTS_LINUX

_SKIP_DIRS_LOWER: set[str] = {d.lower() for d in _SKIP_DIRS_ALL}


# ---------------------------------------------------------------------------
# Scan root resolution
# ---------------------------------------------------------------------------

def _windows_system_drive() -> str:
    win_dir = os.environ.get("SystemRoot", "C:\\Windows")
    return os.path.splitdrive(win_dir)[0].upper().rstrip(":\\")


def get_scan_roots() -> list[str]:
    if CUSTOM_SCAN_ROOTS:
        return list(CUSTOM_SCAN_ROOTS)

    if _OS == "Windows":
        sys_drive = _windows_system_drive()
        roots: list[str] = []
        for letter in string.ascii_uppercase:
            root = f"{letter}:\\"
            if not os.path.exists(root):
                continue
            roots.append(str(Path.home()) if letter.upper() == sys_drive else root)
        return roots

    if _OS == "Darwin":
        roots = [str(Path.home())]
        for vol in Path("/Volumes").iterdir():
            try:
                if vol.is_dir() and not vol.is_symlink():
                    if os.stat(vol).st_dev != os.stat("/").st_dev:
                        roots.append(str(vol))
            except OSError:
                pass
        return roots

    return [str(Path.home())]


# ---------------------------------------------------------------------------
# Venv detection by content — not by name
# ---------------------------------------------------------------------------

def _is_venv(dirpath: str, dirname: str) -> bool:
    """
    Returns True if the given subdirectory is a Python virtual environment.

    Detection method: check for pyvenv.cfg at the directory root.
    This file is created by Python's venv module for every environment
    regardless of what the directory is named.

    This catches: venv, .venv, env, myenv, myenv311, projenv, aqienv,
    voiceagent, flask_env, and any other custom-named environment.
    """
    marker = os.path.join(dirpath, dirname, "pyvenv.cfg")
    return os.path.isfile(marker)


# ---------------------------------------------------------------------------
# Two-layer path filtering (imported by syncer.py)
# ---------------------------------------------------------------------------

def is_dir_skipped(name: str) -> bool:
    """
    Skip a directory if:
      1. Its name starts with a dot (hidden/system dirs like .cursor, .cache)
      2. It is in the SKIP_DIRS set (case-insensitive)
    Note: venv detection is done separately via _is_venv() during the walk
    because it requires checking the filesystem, not just the name.
    """
    if name.startswith("."):
        return True
    return name.lower() in _SKIP_DIRS_LOWER


def is_path_skipped(filepath: str) -> bool:
    n = filepath.lower().replace("\\", "/")
    return any(frag in n for frag in _SKIP_FRAGS)


# ---------------------------------------------------------------------------
# Drive label — single uppercase letter for reliable EQ matching
# ---------------------------------------------------------------------------

def _drive_label(filepath: str) -> str:
    """
    Returns a single uppercase letter: "D", "C", "E" on Windows.
    Returns "/" on Linux/macOS.

    Stored as a single letter so SelfQueryRetriever EQ filters always work.
    The LLM is told to use a single uppercase letter in its filter expression.
    """
    if _OS == "Windows":
        drv, _ = os.path.splitdrive(filepath)
        return drv.upper().rstrip(":\\") if drv else "C"
    return "/"


# ---------------------------------------------------------------------------
# Folder chain (full ancestry from scan root, lowercased)
# ---------------------------------------------------------------------------

def _folder_chain(filepath: str, root: str) -> str:
    fp = filepath.replace("\\", "/")
    rt = root.replace("\\", "/").rstrip("/")

    if fp.lower().startswith(rt.lower()):
        rel = fp[len(rt):].lstrip("/")
    else:
        rel = fp

    parts   = [p for p in rel.split("/") if p]
    folders = parts[:-1]
    return " > ".join(f.lower() for f in folders)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _age_days(ts: float) -> int:
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return (datetime.now(tz=timezone.utc) - dt).days
    except Exception:
        return 99_999


def _ext_score(ext: str) -> float:
    return EXTENSION_PRIORITY.get(ext.lower(), DEFAULT_EXTENSION_PRIORITY)


def _recency_score(mtime: float) -> float:
    age = min(_age_days(mtime), 365)
    return ((365 - age) / 365) * SCORE_WEIGHTS["recency"]


def _depth_score(filepath: str, root: str) -> float:
    try:
        depth = len(Path(os.path.relpath(filepath, root)).parts)
        return max(0.0, SCORE_WEIGHTS["depth"] * (0.65 ** (depth - 1)))
    except ValueError:
        return 0.0


def score_file(filepath: str, ext: str, size_mb: float, mtime: float, root: str) -> float:
    s  = _ext_score(ext)                           * SCORE_WEIGHTS["extension"]
    s += (1.0 if size_mb > 0.05 else 0.0)          * SCORE_WEIGHTS["size"]
    s += _recency_score(mtime)
    s += _depth_score(filepath, root)
    return s


# ---------------------------------------------------------------------------
# Document factory
# ---------------------------------------------------------------------------

def _make_document(filepath: str, stat: os.stat_result, root: str) -> Document:
    filename    = os.path.basename(filepath)
    ext         = os.path.splitext(filename)[1].lstrip(".").lower()
    size_mb     = round(stat.st_size / (1024 * 1024), 4)
    drive       = _drive_label(filepath)
    folder_path = os.path.dirname(filepath)
    folder      = os.path.basename(folder_path)
    created_dt  = datetime.fromtimestamp(stat.st_ctime)
    modified_dt = datetime.fromtimestamp(stat.st_mtime)
    chain       = _folder_chain(filepath, root)

    if chain:
        page_content = f"{filename.lower()} {ext} file in {chain}"
    else:
        page_content = f"{filename.lower()} {ext} file"

    metadata = {
        "filename":    filename,
        "path":        filepath,
        "folder":      folder,
        "folder_path": folder_path,
        "extension":   ext,
        "size_mb":     size_mb,
        "created_at":  created_dt.strftime("%Y-%m-%d"),
        "modified_at": modified_dt.strftime("%Y-%m-%d"),
        "created_ymd": int(created_dt.strftime("%Y%m%d")),
        "modified_ymd": int(modified_dt.strftime("%Y%m%d")),
        "drive":       drive,
    }

    return Document(page_content=page_content, metadata=metadata)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def scan_drives(roots: list[str] | None = None) -> tuple[list[Document], dict[str, int], int]:
    roots = roots or get_scan_roots()

    all_scored: list[tuple[float, Document]] = []
    stats: dict[str, int] = {}
    total = 0

    for root in roots:
        if not os.path.exists(root):
            print(f"[scanner] root not found, skipping: {root}")
            continue

        print(f"[scanner] scanning ({_OS}): {root}")
        count = 0

        for dirpath, dirs, files in os.walk(root, topdown=True):
            # Three-layer directory pruning (order matters for performance):
            #
            # 1. Name-based skip: dot-dirs + known skip-dir names.
            #    Fast O(1) set lookup, eliminates most junk immediately.
            #
            # 2. Venv detection: check for pyvenv.cfg inside each remaining dir.
            #    Catches custom-named venvs like myenv311, projenv, aqienv etc.
            #    One os.path.isfile() call per subdirectory — cheap.
            #
            # 3. Path fragment check on files (layer 2 of the file filter below).

            dirs[:] = [
                d for d in dirs
                if not is_dir_skipped(d)           # layer 1: name + dot check
                and not _is_venv(dirpath, d)        # layer 2: pyvenv.cfg check
            ]

            for filename in files:
                if filename.startswith(SKIP_PREFIX):
                    continue

                ext = os.path.splitext(filename)[1].lstrip(".").lower()
                if ext in SKIP_EXTENSIONS:
                    continue

                full_path = os.path.join(dirpath, filename)

                if is_path_skipped(full_path):
                    continue

                try:
                    stat = os.stat(full_path)
                except (PermissionError, FileNotFoundError, OSError):
                    continue

                doc = _make_document(full_path, stat, root)
                sc  = score_file(
                    full_path, ext,
                    doc.metadata["size_mb"],
                    stat.st_mtime, root,
                )

                all_scored.append((sc, doc))
                count += 1
                total += 1

        stats[root] = count
        print(f"[scanner]   → {count:,} files found in {root}")

    all_scored.sort(key=lambda pair: pair[0], reverse=True)
    docs = [doc for _, doc in all_scored]

    print(f"[scanner] total: {total:,} files across {len(roots)} root(s)")
    return docs, stats, total