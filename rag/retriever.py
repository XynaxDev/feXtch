# =============================================================================
# feXtch — Fast EXtended file feTCH engine
# retriever.py  |  SelfQueryRetriever factory
#
# Author  : Akash Kumar
# Email   : akashkumar.cs27@gmail.com
# GitHub  : github.com/XynaxDev
# Project : feXtch — local AI semantic file search
#
# Extracted from tools.py so tool definitions stay clean.
# Provides: build_retriever(vectorstore, llm) → SelfQueryRetriever | None
# =============================================================================

from __future__ import annotations
from config import RETRIEVER_TOP_K
import re

from langchain_chroma import Chroma
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import (
    AttributeInfo,
    load_query_constructor_runnable,
)


# ---------------------------------------------------------------------------
# Date-safe Chroma translator
# Coerces ISO date strings and LangChain date dicts → YYYYMMDD integers
# so $gt/$lt comparisons never crash on date fields.
# ---------------------------------------------------------------------------

class ChromaDateSafeTranslator(ChromaTranslator):

    @staticmethod
    def _coerce(v):
        if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            return int(v.replace("-", ""))
        if isinstance(v, dict) and "date" in v:
            inner = v["date"]
            if isinstance(inner, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", inner):
                return int(inner.replace("-", ""))
        return v

    def _walk(self, node):
        if isinstance(node, dict):
            return {
                k: (self._coerce(self._walk(v))
                    if k in {"$gt", "$lt", "$gte", "$lte"}
                    else self._walk(v))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [self._walk(i) for i in node]
        return self._coerce(node)

    def visit_operation(self, op):
        return self._walk(super().visit_operation(op))


# ---------------------------------------------------------------------------
# Metadata schema
# Only fields Chroma actually supports filters on (EQ, GT, LT).
# CONTAIN is NOT supported — do not add folder_path_lower or filename_lower.
# ---------------------------------------------------------------------------

METADATA_FIELD_INFO: list[AttributeInfo] = [
    AttributeInfo(
        name="extension",
        description="File extension without dot, always lowercase: tsx, py, pdf, ipynb, md.",
        type="string",
    ),
    AttributeInfo(
        name="drive",
        description=(
            "Single uppercase drive letter. "
            "Windows: 'C' for C drive, 'D' for D drive. "
            "Linux/macOS: '/'. Single letter only — never 'D:\\\\' or 'D:'."
        ),
        type="string",
    ),
    AttributeInfo(
        name="size_mb",
        description="File size in megabytes as a float. Use GT/LT for size filters.",
        type="float",
    ),
    AttributeInfo(
        name="created_ymd",
        description="File creation date as YYYYMMDD integer. Use GT/LT. 20250101 = 2025-01-01.",
        type="float",
    ),
    AttributeInfo(
        name="modified_ymd",
        description="Last-modified date as YYYYMMDD integer. Use GT/LT. 20250315 = 2025-03-15.",
        type="float",
    ),
]

_DOCUMENT_CONTENTS = (
    "File and folder names with metadata on the user's local computer. "
    "Each document represents one file. Page content is lowercased and contains "
    "the filename plus the full folder ancestry chain separated by ' > '. "
    "Example: 'app.py py file in myproject > backend > routes'. "
    "Folder names are embedded — semantic search finds files by project name. "
    "Only use metadata filters for: extension (EQ), drive (EQ single letter), "
    "size_mb (GT/LT), created_ymd (GT/LT), modified_ymd (GT/LT)."
)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_retriever(vectorstore: Chroma, llm, top_k: int = RETRIEVER_TOP_K) -> SelfQueryRetriever | None:
    """
    Build a SelfQueryRetriever bound to the given Chroma vectorstore and LLM.
    Returns None if construction fails — callers should fall back to
    plain similarity_search in that case.

    The retriever decomposes a natural language query into:
      1. A semantic search string  → vector similarity in Chroma
      2. A structured metadata filter → applied on top (extension, drive, size, date)
    """
    if llm is None:
        return None
    try:
        qc = load_query_constructor_runnable(
            llm=llm,
            document_contents=_DOCUMENT_CONTENTS,
            attribute_info=METADATA_FIELD_INFO,
        )
        return SelfQueryRetriever(
            query_constructor=qc,
            vectorstore=vectorstore,
            structured_query_translator=ChromaDateSafeTranslator(),
            search_kwargs={"k": top_k},
            verbose=False,
        )
    except Exception:
        return None