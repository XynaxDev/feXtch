# =============================================================================
# feXtch — Fast EXtended file feTCH engine
# tools.py  |  LangChain tools for the file search agent
#
# Author  : Akash Kumar
# Email   : akashkumar.cs27@gmail.com
# GitHub  : github.com/XynaxDev
# Project : feXtch — local AI semantic file search
#
# Each tool is a self-contained operation.
# The agent in generator.py decides which tool(s) to call per query.
#
# Count accuracy note
# ────────────────────
# similarity_search(k=N) returns at most N docs — useless for counting.
# db.get(where={...}, include=["metadatas"]) scans the full collection
# with exact metadata filters and returns ALL matching docs.
# This is the correct API for count/aggregate queries.
# =============================================================================

from __future__ import annotations
import re
from collections import Counter
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from retriever import build_retriever
from config import RETRIEVER_TOP_K, RESULTS_SHOW, FOLDER_SHOW, APPROX_MARGIN


# ---------------------------------------------------------------------------
# Helpers shared across tools
# ---------------------------------------------------------------------------

# File type group map — when user says "images", search all image extensions.
# No hardcoded folder names or paths — purely extension groups.
FILE_TYPE_GROUPS: dict[str, list[str]] = {
    "image":    ["jpg", "jpeg", "png", "gif", "webp", "bmp", "svg", "tiff", "heic", "raw", "ico"],
    "images":   ["jpg", "jpeg", "png", "gif", "webp", "bmp", "svg", "tiff", "heic", "raw", "ico"],
    "photo":    ["jpg", "jpeg", "png", "heic", "raw", "webp"],
    "photos":   ["jpg", "jpeg", "png", "heic", "raw", "webp"],
    "picture":  ["jpg", "jpeg", "png", "gif", "webp", "bmp"],
    "pictures": ["jpg", "jpeg", "png", "gif", "webp", "bmp"],
    "video":    ["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v"],
    "videos":   ["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v"],
    "document": ["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "txt", "md", "odt"],
    "documents":["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "txt", "md", "odt"],
    "audio":    ["mp3", "wav", "flac", "aac", "ogg", "m4a", "wma"],
    "notebook": ["ipynb"],
    "notebooks":["ipynb"],
    "script":   ["py", "js", "ts", "sh", "bat", "ps1"],
    "scripts":  ["py", "js", "ts", "sh", "bat", "ps1"],
    "archive":  ["zip", "rar", "tar", "gz", "7z", "bz2"],
    "archives": ["zip", "rar", "tar", "gz", "7z", "bz2"],
}


def _resolve_extension(extension: str | None, query: str) -> list[str] | None:
    """
    Given an extension filter and the query string, return a list of extensions to search.
    If the query contains a file-type group word (image, video, etc.) and no explicit
    extension was provided, expand to the full group.
    Returns None if no extension filter should be applied.
    """
    if extension:
        # explicit extension — use it directly
        ext = extension.lower().lstrip(".")
        # but if it's a group name itself, expand it
        if ext in FILE_TYPE_GROUPS:
            return FILE_TYPE_GROUPS[ext]
        return [ext]
    # scan query tokens for group names
    tokens = re.sub(r"[^a-z\s]", " ", query.lower()).split()
    for tok in tokens:
        if tok in FILE_TYPE_GROUPS:
            return FILE_TYPE_GROUPS[tok]
    return None


def _normalise_parts(path: str) -> list[str]:
    return [p for p in path.replace("\\", "/").split("/") if p]


def _build_where(
    drive: str | None = None,
    extension: str | None = None,
    min_size_mb: float | None = None,
    max_size_mb: float | None = None,
    modified_after: int | None = None,
    modified_before: int | None = None,
) -> dict | None:
    """Build a Chroma where-filter dict from optional criteria."""
    clauses = []

    if drive:
        # accept "D", "d", "D:", "D:\\" — normalise to single uppercase letter
        letter = re.sub(r"[:\\/\s]", "", drive).upper()
        if letter:
            clauses.append({"drive": {"$eq": letter}})

    if extension:
        clauses.append({"extension": {"$eq": extension.lower().lstrip(".")}})

    if min_size_mb is not None:
        clauses.append({"size_mb": {"$gte": min_size_mb}})

    if max_size_mb is not None:
        clauses.append({"size_mb": {"$lte": max_size_mb}})

    if modified_after is not None:
        clauses.append({"modified_ymd": {"$gte": modified_after}})

    if modified_before is not None:
        clauses.append({"modified_ymd": {"$lte": modified_before}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _tok_matches_component(tok: str, part: str) -> bool:
    """
    Check if a search token matches a path component at word boundaries.
    Separators: hyphen, underscore, ampersand, dot, space.
    'isro' matches 'ISRO&DRDO' but NOT 'unisro'.
    'dataset' matches 'my-dataset' but NOT 'datasets'.
    """
    t = tok.lower()
    p = part.lower()
    if p == t:
        return True
    return bool(re.search(rf"(^|[-_&.\s]){re.escape(t)}([-_&.\s]|$)", p)) or (len(t) >= 4 and (p.startswith(t) or p.endswith(t)))


def _fmt_file(m: dict) -> str:
    return (
        f"• {m.get('filename')}\n"
        f"  path     : {m.get('path')}\n"
        f"  drive    : {m.get('drive')}\n"
        f"  ext      : {m.get('extension')}\n"
        f"  size     : {m.get('size_mb')} MB\n"
        f"  modified : {m.get('modified_at')}\n"
    )


def _top_recent_folders(metas: list[dict], n: int = 10) -> str:
    """
    Return top N recently active folders, diverse across projects.
    Groups all subdirectories under the same top-level root so UniGPT does not
    fill all 10 slots with its own subdirs.
    Strategy: pick the most recently active folder per unique top-level root,
    then fill remaining slots from other roots in recency order.
    """
    import re as _re

    def _top_root(fp: str) -> str:
        parts = [p for p in fp.replace("\\", "/").split("/") if p]
        if len(parts) >= 2 and ":" in parts[0]:
            return parts[0] + "\\" + parts[1]
        return parts[0] if parts else fp

    # build per-folder recency
    folder_recency: dict[str, dict] = {}
    for m in metas:
        fp  = m.get("folder_path", "")
        ymd = m.get("modified_ymd", 0) or 0
        drv = m.get("drive", "")
        if not fp: continue
        if fp not in folder_recency or ymd > folder_recency[fp]["ymd"]:
            folder_recency[fp] = {"path": fp, "drive": drv, "ymd": ymd,
                                   "modified_at": m.get("modified_at", ""), "count": 0,
                                   "root": _top_root(fp)}
        folder_recency[fp]["count"] += 1

    all_sorted = sorted(folder_recency.values(), key=lambda x: x["ymd"], reverse=True)
    if not all_sorted: return ""

    # Pick one representative per top-level root (most recently active sub)
    # then fill remaining slots from other roots in order
    seen_roots: set[str] = set()
    diverse: list[dict] = []
    leftover: list[dict] = []
    for f in all_sorted:
        if f["root"] not in seen_roots:
            seen_roots.add(f["root"])
            diverse.append(f)
        else:
            leftover.append(f)

    # fill to n if diverse < n (add remaining different roots first, then leftovers)
    result = (diverse + leftover)[:n]

    lines = [f"top {len(result)} recently active folders (across {len(seen_roots)} projects):"]
    for f in result:
        lines.append(f"  \U0001f4c1 {f['path']}")
        lines.append(f"     drive: {f['drive']}  last modified: {f['modified_at']}  files here: {f['count']}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Tool factory — returns tools bound to a specific db + vectorstore
# ---------------------------------------------------------------------------

def make_tools(db: Chroma, vectorstore: Chroma, llm=None):
    """
    Create all feXtch tools bound to the given Chroma instance.
    Returns a list of LangChain tool objects ready for an agent.
    """

    # SelfQueryRetriever — built once, reused by search_files
    from config import RETRIEVER_TOP_K
    _retriever = build_retriever(vectorstore, llm, top_k=RETRIEVER_TOP_K)

    @tool
    def search_files(
        query: str,
        extension: str | None = None,
        drive: str | None = None,
        min_size_mb: float | None = None,
        modified_after: int | None = None,
    ) -> str:
        """
        Search for files using semantic similarity.
        Use for: finding files by topic, technology, or name pattern.
        Do NOT use for counting — use count_files instead.

        Args:
            query         : natural language search string (e.g. "machine learning notebook")
            extension     : filter by file type, e.g. "py", "pdf", "tsx" (optional)
            drive         : single drive letter, e.g. "D" or "C" (optional — omit to search all drives)
            min_size_mb   : minimum file size in MB (optional)
            modified_after: YYYYMMDD integer, e.g. 20250101 (optional)
        """
        safe_q = re.sub(r"[&|!@#$%^*()+={};:<>?]", " ", query.lower())
        safe_q = re.sub(r"\s+", " ", safe_q).strip() or query.lower()

        # Expand group words like "images" → [jpg,png,gif,...]
        ext_list = _resolve_extension(extension, safe_q)

        # Approx size detection — "approx 400 mb" → 300–500 MB range
        # Detects: approx, approximately, around, about, ~, near, roughly
        max_size_mb_approx: float | None = None
        approx_match = re.search(
            r"(approx|approximately|around|about|~|near|roughly)[^\d]*(\d+(?:\.\d+)?)\s*(mb|gb|kb)?",
            safe_q
        )
        if approx_match:
            val  = float(approx_match.group(2))
            unit = (approx_match.group(3) or "mb").lower()
            if unit == "gb": val *= 1024
            if unit == "kb": val /= 1024
            margin = val * APPROX_MARGIN  # ±25%
            min_size_mb = round(val - margin, 2)
            max_size_mb_approx = round(val + margin, 2)
        else:
            max_size_mb_approx = None

        def _do_search(ext_filter):
            where = _build_where(drive, ext_filter, min_size_mb, max_size_mb_approx, modified_after)
            parts = [safe_q]
            if ext_filter:     parts.append(f"extension {ext_filter}")
            if drive:          parts.append(f"on drive {drive}")
            if modified_after: parts.append(f"modified after {modified_after}")
            full_q = " ".join(parts)
            try:
                # Bypass SelfQueryRetriever when we have a Python-computed range
                # (approx size, multi-extension group) — it would overwrite our filter.
                # Use it only for plain semantic queries with no overriding Python filter.
                use_sqr = (
                    _retriever is not None
                    and max_size_mb_approx is None       # no approx range
                    and not (ext_list and len(ext_list) > 1)  # not multi-ext group
                )
                if use_sqr:
                    return _retriever.invoke(full_q)
                return vectorstore.similarity_search(safe_q, k=RETRIEVER_TOP_K, filter=where) if where \
                    else vectorstore.similarity_search(safe_q, k=RETRIEVER_TOP_K)
            except Exception:
                try:
                    return vectorstore.similarity_search(safe_q, k=RETRIEVER_TOP_K, filter=where) if where \
                        else vectorstore.similarity_search(safe_q, k=RETRIEVER_TOP_K)
                except Exception:
                    return []

        # For group queries search each extension and merge results
        if ext_list and len(ext_list) > 1:
            all_results: list = []
            seen_p: set[str] = set()
            for ext in ext_list:
                for doc in _do_search(ext):
                    p = doc.metadata.get("path", "")
                    if p not in seen_p:
                        seen_p.add(p)
                        all_results.append(doc)
            results = all_results
        else:
            results = _do_search(ext_list[0] if ext_list else None)


        if not results:
            return "(no files found)"

        # deduplicate + exact-match boost
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in results:
            p = doc.metadata.get("path", "")
            if p not in seen:
                seen.add(p)
                unique.append(doc)

        tokens = [t for t in safe_q.split() if len(t) >= 3]

        def _result_score(doc: Document) -> tuple:
            m   = doc.metadata
            exact = sum(1 for t in tokens if t in doc.page_content)
            path  = m.get("path", "")
            depth = path.replace("\\", "/").count("/")
            size  = m.get("size_mb", 0) or 0
            # For approx queries: rank by PROXIMITY to target value
            # Files closest to target_size come first
            if max_size_mb_approx is not None and min_size_mb is not None:
                target = (min_size_mb + max_size_mb_approx) / 2
                proximity = -abs(size - target)  # negative so closer = higher sort key
                return (exact, proximity)
            return (exact, -depth, size)

        unique.sort(key=_result_score, reverse=True)
        return "\n".join(_fmt_file(d.metadata) for d in unique[:RESULTS_SHOW])


    @tool
    def count_files(
        folder_name: str | None = None,
        extension: str | None = None,
        drive: str | None = None,
        min_size_mb: float | None = None,
        modified_after: int | None = None,
        count_folders: bool = False,
    ) -> str:
        """
        Count files or folders accurately using exact database scan.
        Use for: "how many files", "how many folders", "total", "count" queries.
        Always returns exact numbers.

        Args:
            folder_name   : filter to paths containing this name (e.g. "ISRO&DRDO")
            extension     : filter by file type e.g. "pdf", "py" (optional)
            drive         : single drive letter e.g. "D" (optional — omit for all drives)
            min_size_mb   : minimum file size filter (optional)
            modified_after: YYYYMMDD integer filter (optional)
            count_folders : if True, count unique top-level project folders, not files
        """
        where = _build_where(drive, extension, min_size_mb, None, modified_after)

        try:
            result = db.get(where=where, include=["metadatas"]) if where                 else db.get(include=["metadatas"])
            metas = result.get("metadatas", []) or []
        except Exception as e:
            return f"(count error: {e})"

        if folder_name:
            tokens = [t.lower() for t in re.split(r"[&|\s]+", folder_name) if t]
            name_lower = folder_name.lower()

            def _exact_folder_match(m):
                """Check if any path component is exactly the folder_name (case-insensitive)."""
                fp_parts = _normalise_parts(m.get("folder_path", "") or m.get("path", ""))
                return any(part.lower() == name_lower for part in fp_parts)

            def _token_folder_match(m):
                """Fallback: check if all tokens match path components."""
                fp_parts = _normalise_parts(m.get("folder_path", "") or m.get("path", ""))
                return any(_tok_matches_component(tok, part)
                           for tok in tokens for part in fp_parts)

            # Try exact match first (e.g. user typed 'ISRO&DRDO' and there's a folder named 'ISRO&DRDO')
            exact_metas = [m for m in metas if _exact_folder_match(m)]
            metas = exact_metas if exact_metas else [m for m in metas if _token_folder_match(m)]

        if count_folders:
            top_level: set[str] = set()
            all_unique: set[str] = set()
            for m in metas:
                fp = m.get("folder_path", "")
                if not fp: continue
                all_unique.add(fp)
                parts = [p for p in fp.replace("\\", "/").split("/") if p]
                if len(parts) >= 2 and ":" in parts[0]:
                    top_level.add(parts[0] + "\\" + parts[1])
                elif len(parts) >= 1:
                    top_level.add(parts[0])

            if not top_level:
                return "(no folders found)"

            scope_parts_f = []
            if folder_name: scope_parts_f.append(f"in {repr(folder_name)}")
            if drive:       scope_parts_f.append(f"on drive {drive}")
            scope_label = " ".join(scope_parts_f)

            header = f"Found {len(top_level):,} top-level folders"
            if scope_label: header += f" {scope_label}"
            scope_context = f" {scope_label}" if scope_label else ""
            lines_out = [
                header + ":",
                f"({len(all_unique):,} total subdirectories{scope_context})",
                "",
                f"Top 10 most recently active{scope_context}:",
            ]
            folder_summary = _top_recent_folders(metas, n=FOLDER_SHOW)
            if folder_summary:
                summary_lines = folder_summary.split("\n")
                rest = summary_lines[1:] if summary_lines and summary_lines[0].startswith("top") else summary_lines
                lines_out.extend(rest)
            return "\n".join(lines_out)


        # File count
        total  = len(metas)
        if total == 0:
            scope = " ".join(filter(None, [
                f".{extension}" if extension else None,
                f"in \'{folder_name}\'" if folder_name else None,
                f"on drive {drive}" if drive else None,
            ]))
            return f"(no {scope or ''} files found)"

        all_dirs = len({m.get("folder_path", "") for m in metas if m.get("folder_path")})
        by_ext   = Counter(m.get("extension", "") for m in metas if m.get("extension"))
        by_drv   = Counter(m.get("drive", "") for m in metas if m.get("drive"))

        scope_parts = []
        if extension:    scope_parts.append(f".{extension} files")
        else:            scope_parts.append("files (all types)")
        if folder_name:  scope_parts.append(f"in \'{folder_name}\'")
        if drive:        scope_parts.append(f"on drive {drive}")

        # Sort by most recently modified
        SHOW = RESULTS_SHOW
        metas_sorted = sorted(metas, key=lambda m: m.get("modified_ymd", 0) or 0, reverse=True)

        scope_str = " ".join(scope_parts)
        lines = [f"Found {total:,} {scope_str}:"]

        if count_folders:
            # user asked about folders — show folder summary only
            folder_summary = _top_recent_folders(metas, n=FOLDER_SHOW)
            if folder_summary:
                lines.append("")
                lines.append(folder_summary)
        else:
            # user asked about files — show top 10 files + diverse folder summary
            lines.append(f"\nTop {min(SHOW, total)} recently modified:")
            for m in metas_sorted[:SHOW]:
                lines.append(_fmt_file(m).rstrip())
            if total > SHOW:
                lines.append(f"  ... and {total - SHOW:,} more not shown")
            # folder summary — diverse across projects, not just the same project repeated
            folder_summary = _top_recent_folders(metas, n=FOLDER_SHOW)
            if folder_summary:
                lines.append("")
                lines.append(folder_summary)

        return "\n".join(lines)


    @tool
    def find_folder(
        name: str,
        drive: str | None = None,
    ) -> str:
        """
        Find where a named project or directory is located.
        Use for: "where is X", "locate X folder", "find X directory", or bare project names.
        Returns results grouped: top-level matching dirs first, then sub-directories under them.

        Args:
            name  : folder or project name to find (e.g. "flask-blog", "ISRO&DRDO")
            drive : limit search to one drive letter e.g. "D" (optional)
        """
        where = _build_where(drive)

        try:
            result = db.get(where=where, include=["metadatas"]) if where                 else db.get(include=["metadatas"])
            metas = result.get("metadatas", []) or []
        except Exception as e:
            return f"(error: {e})"

        tokens = [t.lower() for t in re.split(r"[&|\s_-]+", name) if t]

        # collect unique folder paths where any path component matches a token
        folder_map: dict[str, dict] = {}
        for m in metas:
            fp = m.get("folder_path", "")
            if not fp:
                continue
            parts = _normalise_parts(fp)
            for part in parts:
                if any(_tok_matches_component(tok, part) for tok in tokens):
                    if fp not in folder_map:
                        folder_map[fp] = {
                            "path":     fp,
                            "drive":    m.get("drive", ""),
                            "examples": [],
                        }
                    if len(folder_map[fp]["examples"]) < 3:
                        folder_map[fp]["examples"].append(m.get("filename", ""))
                    break

        if not folder_map:
            for m in metas:
                fp  = m.get("folder_path", "")
                if not fp:
                    continue
                # even in fallback, check word boundaries to avoid false positives
                parts_fb = _normalise_parts(fp)
                if any(_tok_matches_component(tok, part) for tok in tokens for part in parts_fb):
                    if fp not in folder_map:
                        folder_map[fp] = {"path": fp, "drive": m.get("drive", ""), "examples": [m.get("filename", "")]}

        if not folder_map and vectorstore is not None:
            # Semantic fallback — search page_content directly
            # page_content stores the lowercased folder ancestry chain
            # so "meewwww" will match "somefile.txt txt file in meewwww"
            # This handles any folder name including nonsense words
            try:
                safe_q = re.sub(r"[&|!@#$%^*()+={};:<>?]", " ", name.lower()).strip()
                sem_results = vectorstore.similarity_search(safe_q, k=RETRIEVER_TOP_K)
                for doc in sem_results:
                    fp  = doc.metadata.get("folder_path", "")
                    drv = doc.metadata.get("drive", "")
                    fn  = doc.metadata.get("filename", "")
                    if not fp: continue
                    parts_s = _normalise_parts(fp)
                    # only add if the folder name appears literally in the path
                    name_lower_s = name.lower()
                    if any(name_lower_s in part.lower() or part.lower() in name_lower_s
                           for part in parts_s):
                        if fp not in folder_map:
                            folder_map[fp] = {"path": fp, "drive": drv, "examples": []}
                        if len(folder_map[fp]["examples"]) < 3:
                            folder_map[fp]["examples"].append(fn)
            except Exception:
                pass


        if not folder_map:
            return (
                f"No folder named '{name}' found in the index.\n"
                "This usually means the folder is empty — feXtch indexes FILES, not folders.\n"
                "Add at least one file inside it, run syncer.py, then search again."
            )

        all_paths = list(folder_map.keys())

        def _norm(p):
            return re.sub(r"[/\\\\]+", "/", p).lower().rstrip("/")

        def _is_ancestor(a, d):
            an, dn = _norm(a), _norm(d)
            return an != dn and dn.startswith(an + "/")

        # Separate root-level matches from nested ones
        roots = [p for p in all_paths if not any(_is_ancestor(other, p) for other in all_paths)]
        # Sort: exact name match first, then by path
        roots.sort(key=lambda p: (0 if any(tok == _norm(p).split("/")[-1] for tok in tokens) else 1, p.lower()))

        lines = []
        for root_path in roots:
            f = folder_map[root_path]
            ex = ", ".join(f["examples"]) if f["examples"] else "—"
            lines.append(f"📁 {root_path}")
            lines.append(f"   drive: {f['drive']}  |  sample files: {ex}")

            # show children (sub-dirs of this root that also matched)
            children = sorted([p for p in all_paths if _is_ancestor(root_path, p)])
            if children:
                lines.append(f"   subdirectories also matched ({len(children)}):")
                for child in children[:4]:
                    rel = child[len(root_path):].lstrip("\\/")
                    lines.append(f"     └─ {rel}")
                if len(children) > 4:
                    lines.append(f"     └─ ... and {len(children)-4} more")
            lines.append("")

        return "\n".join(lines).rstrip()


    @tool
    def find_folders_with_subfolder(
        subfolder_name: str,
        drive: str | None = None,
    ) -> str:
        """
        Find all parent directories that contain a specific named subdirectory.
        Use for: "folders containing src", "projects with a backend folder", "which dirs have tests".

        Args:
            subfolder_name: name of the subfolder to look for (e.g. "src", "tests", "backend")
            drive         : limit to one drive (optional)
        """
        where = _build_where(drive)

        try:
            result = db.get(where=where, include=["metadatas"]) if where \
                else db.get(include=["metadatas"])
            metas = result.get("metadatas", []) or []
        except Exception as e:
            return f"(error: {e})"

        sub_lower = subfolder_name.lower()
        parents: dict[str, dict] = {}

        for m in metas:
            path  = m.get("path", "")
            drv   = m.get("drive", "")
            parts = _normalise_parts(path)

            for i, part in enumerate(parts[:-1]):
                if part.lower() == sub_lower and i > 0:
                    parent_parts = parts[:i]
                    if ":" in parts[0]:
                        parent = parts[0] + "\\" + "\\".join(parent_parts[1:])
                    else:
                        parent = "/" + "/".join(parent_parts)

                    if parent not in parents:
                        parents[parent] = {"path": parent, "drive": drv, "examples": []}
                    if len(parents[parent]["examples"]) < 3:
                        parents[parent]["examples"].append(m.get("filename", ""))
                    break

        if not parents:
            return f"(no folders found containing a '{subfolder_name}' subfolder)"

        lines = []
        for p in sorted(parents.values(), key=lambda x: x["path"].lower()):
            ex = ", ".join(p["examples"]) if p["examples"] else "—"
            lines.append(f"📁 {p['path']}\n   drive: {p['drive']}  contains: {subfolder_name}/  examples: {ex}")
        return "\n".join(lines)


    @tool
    def most_recent_items(
        drive: str | None = None,
        extension: str | None = None,
        item_type: str = "file",
        modified_after: int | None = None,
        modified_before: int | None = None,
        created_after: int | None = None,
        created_before: int | None = None,
    ) -> str:
        """
        Find files or folders filtered by date range.
        Use for: "recently modified", "made in september 2025", "created last month",
                 "folders from 2024", "files changed this week".

        Args:
            drive          : single drive letter (optional)
            extension      : file type filter (optional)
            item_type      : "file" or "folder"
            modified_after : YYYYMMDD — files modified on or after this date
            modified_before: YYYYMMDD — files modified on or before this date
            created_after  : YYYYMMDD — files created on or after this date
            created_before : YYYYMMDD — files created on or before this date

        For "september 2025": modified_after=20250901, modified_before=20250930
        For "2024": modified_after=20240101, modified_before=20241231
        For "recently" with no date: leave both None to get the most recent items.
        """
        where = _build_where(drive, extension,
                             modified_after=modified_after,
                             modified_before=modified_before)

        # created_ymd filters applied in Python (Chroma supports $gte/$lte)
        created_clauses = []
        if created_after:  created_clauses.append(("$gte", created_after))
        if created_before: created_clauses.append(("$lte", created_before))

        try:
            result = db.get(where=where, include=["metadatas"]) if where \
                else db.get(include=["metadatas"])
            metas = result.get("metadatas", []) or []
        except Exception as e:
            return f"(error: {e})"

        # Apply created_ymd filters in Python
        for op, val in created_clauses:
            if op == "$gte":
                metas = [m for m in metas if (m.get("created_ymd") or 0) >= val]
            else:
                metas = [m for m in metas if (m.get("created_ymd") or 0) <= val]

        if not metas:
            scope = []
            if modified_after:  scope.append(f"after {str(modified_after)[:4]}-{str(modified_after)[4:6]}-{str(modified_after)[6:]}")
            if modified_before: scope.append(f"before {str(modified_before)[:4]}-{str(modified_before)[4:6]}-{str(modified_before)[6:]}")
            return f"(no {item_type}s found{(' ' + ' and '.join(scope)) if scope else ''})"

        SHOW = RESULTS_SHOW

        # Build scope label for header
        scope_parts = []
        if modified_after and modified_before:
            # format as month range if same month
            after_s  = str(modified_after)
            before_s = str(modified_before)
            if after_s[4:6] == before_s[4:6] and after_s[:4] == before_s[:4]:
                import calendar
                month_name = calendar.month_name[int(after_s[4:6])]
                scope_parts.append(f"from {month_name} {after_s[:4]}")
            else:
                scope_parts.append(f"between {after_s[:4]}-{after_s[4:6]}-{after_s[6:]} and {before_s[:4]}-{before_s[4:6]}-{before_s[6:]}")
        elif modified_after:
            scope_parts.append(f"after {str(modified_after)[:4]}-{str(modified_after)[4:6]}-{str(modified_after)[6:]}")
        elif modified_before:
            scope_parts.append(f"before {str(modified_before)[:4]}-{str(modified_before)[4:6]}-{str(modified_before)[6:]}")
        if drive:     scope_parts.append(f"on drive {drive}")
        scope_label = " ".join(scope_parts)

        if item_type == "folder":
            # Group by folder_path, use the most recent file inside as the folder date
            folder_recency: dict[str, dict] = {}
            for m in metas:
                fp  = m.get("folder_path", "")
                ymd = m.get("modified_ymd", 0) or 0
                drv = m.get("drive", "")
                fn  = m.get("filename", "")
                if not fp: continue
                if fp not in folder_recency or ymd > folder_recency[fp]["ymd"]:
                    folder_recency[fp] = {"path": fp, "drive": drv, "ymd": ymd,
                                           "modified_at": m.get("modified_at",""), "example": fn, "count": 0}
                folder_recency[fp]["count"] += 1

            # Apply diversity — one folder per top-level project
            all_sorted = sorted(folder_recency.values(), key=lambda x: x["ymd"], reverse=True)
            seen_roots: set[str] = set()
            diverse: list[dict] = []
            leftover: list[dict] = []
            for f in all_sorted:
                parts_r = _normalise_parts(f["path"])
                root = (parts_r[0] + "\\" + parts_r[1]) if len(parts_r) >= 2 and ":" in parts_r[0] else parts_r[0]
                if root not in seen_roots:
                    seen_roots.add(root)
                    diverse.append(f)
                else:
                    leftover.append(f)
            result_folders = (diverse + leftover)[:SHOW]

            total = len(folder_recency)
            header = f"Found {total:,} folders {scope_label}:" if scope_label else f"Top {min(SHOW, total)} recently active folders:"
            lines = [header, ""]
            for f in result_folders:
                lines.append(f"  📁 {f['path']}")
                lines.append(f"     drive: {f['drive']}  last modified: {f['modified_at']}  files here: {f['count']}")
            if total > SHOW:
                lines.append(f"  ... and {total - SHOW:,} more folders not shown")
            return "\n".join(lines)

        else:  # file
            all_sorted = sorted(metas, key=lambda m: m.get("modified_ymd", 0) or 0, reverse=True)
            total = len(all_sorted)
            shown = all_sorted[:SHOW]

            header = f"Found {total:,} files {scope_label}:" if scope_label else f"Top {min(SHOW, total)} recently modified files:"
            lines = [header]
            for m in shown:
                lines.append(_fmt_file(m).rstrip())
            if total > SHOW:
                lines.append(f"  ... and {total - SHOW:,} more not shown")
            folder_summary = _top_recent_folders(metas, n=FOLDER_SHOW)
            if folder_summary:
                lines.append("")
                lines.append(folder_summary)
            return "\n".join(lines)

    return [search_files, count_files, find_folder, find_folders_with_subfolder, most_recent_items]