# =============================================================================
# feXtch — Fast EXtended file feTCH engine
# intent.py  |  LLM-powered query intent detection via Pydantic structured output
#
# Author  : Akash Kumar
# Email   : akashkumar.cs27@gmail.com
# GitHub  : github.com/XynaxDev
# Project : feXtch — local AI semantic file search
# =============================================================================

from __future__ import annotations
import warnings
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
)
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class QueryIntent(BaseModel):
    mode: str = Field(
        description=(
            "One of four values:\n"
            "  'file'             — user wants specific files\n"
            "  'folder'           — user wants folder/directory locations or project paths\n"
            "  'subfolder_search' — user wants folders that CONTAIN a specific subfolder\n"
            "  'count'            — user wants a count/total/number of files or folders\n"
        )
    )

    semantic_query: str = Field(
        description=(
            "Clean lowercase search string for vector similarity. "
            "Keep the meaningful subject as-is. "
            "Do NOT rephrase or expand — if the query is 'UnivGPT', semantic_query is 'univgpt'."
        )
    )

    search_terms: list[str] = Field(
        default_factory=list,
        description="Key lowercase tokens from the query for exact-match boosting."
    )

    file_extension: str | None = Field(
        default=None,
        description="Extension without dot if user filtered by type: 'py', 'tsx', 'pdf'. Null otherwise."
    )

    drive: str | None = Field(
        default=None,
        description="Single uppercase drive letter if specified: 'D', 'C'. Null if not mentioned."
    )

    subfolder_name: str | None = Field(
        default=None,
        description="Only for subfolder_search mode: the subfolder name to find inside parents."
    )

    modified_after: int | None = Field(
        default=None,
        description="YYYYMMDD integer if user specified a date after which files were modified."
    )

    modified_before: int | None = Field(
        default=None,
        description="YYYYMMDD integer if user specified a date before which files were modified."
    )

    min_size_mb: float | None = Field(default=None)
    max_size_mb: float | None = Field(default=None)


# ---------------------------------------------------------------------------
# System prompt — includes bare-name example
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = """\
You are a query analyser for a local filesystem search engine.

Given a natural language query, extract structured search intent.

Mode selection rules:
- mode = "file"             → user wants specific FILES (filenames, extensions, document content)
- mode = "folder"           → user wants to know WHERE a folder/project/directory is located
- mode = "subfolder_search" → user wants FOLDERS that CONTAIN a specific named subdirectory
- mode = "count"            → user wants a NUMBER — "how many", "count", "total", "number of"

IMPORTANT — bare name rule:
If the query is a single word or short name with no file extension (e.g. "UnivGPT", "flask-blog",
"MyProject"), treat it as mode="folder". The user is asking where that project/folder is.

Examples:
  "find app.py"                           → mode=file,   semantic_query="app.py"
  "UnivGPT"                               → mode=folder, semantic_query="univgpt"
  "flask-blog"                            → mode=folder, semantic_query="flask-blog"
  "where is my flask project"             → mode=folder, semantic_query="flask project"
  "all folders containing a src folder"   → mode=subfolder_search, subfolder_name="src"
  "tsx files in my frontend project"      → mode=file,   file_extension="tsx"
  "projects that have a tests directory"  → mode=subfolder_search, subfolder_name="tests"
  "large pdf files on D drive"            → mode=file,   file_extension="pdf", drive="D"
  "python notebooks from 2025"            → mode=file,   file_extension="ipynb", modified_after=20250101
  "how many pdf files in ISRO&DRDO"       → mode=count,  file_extension="pdf",   semantic_query="isro drdo"
  "how many files are in my D drive"      → mode=count,  drive="D"
  "count all python files"                → mode=count,  file_extension="py"
  "how many folders on D drive"           → mode=count,  drive="D"
  "total tsx files in myproject"          → mode=count,  file_extension="tsx",   semantic_query="myproject"

Return ONLY valid JSON matching the schema. No explanation, no markdown.
"""


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def detect_intent(question: str, llm) -> QueryIntent:
    """
    Use the LLM to extract structured intent. Returns a QueryIntent object.
    Falls back to a broad folder search on bare names, file search otherwise.
    """
    try:
        structured_llm = llm.with_structured_output(QueryIntent)
        result = structured_llm.invoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=question),
        ])
        # with_structured_output may return a dict on some providers — handle both
        if isinstance(result, dict):
            return QueryIntent.model_validate(result)
        return result
    except Exception:
        import re
        # bare name with no extension and no spaces → folder intent
        q = question.strip()
        count_words = {"how many", "count", "total", "number of", "how much"}
        if any(cw in q.lower() for cw in count_words):
            return QueryIntent(
                mode="count",
                semantic_query=q.lower(),
                search_terms=q.lower().split(),
            )
        if re.match(r'^[\w-]+$', q) and '.' not in q:
            return QueryIntent(
                mode="folder",
                semantic_query=question.lower().strip(),
                search_terms=[question.lower().strip()],
            )
        return QueryIntent(
            mode="file",
            semantic_query=question.lower(),
            search_terms=question.lower().split(),
        )