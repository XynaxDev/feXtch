# =============================================================================
# feXtch — Fast EXtended file feTCH engine
# generator.py  |  Tool-calling agent + history-aware chatbot
#
# Author  : Akash Kumar
# Email   : akashkumar.cs27@gmail.com
# GitHub  : github.com/akashkumar
# Project : feXtch — local AI semantic file search
#
# Architecture
# ────────────
# Each user query goes through:
#   1. Condense (if follow-up) — rewrites using history, never adds context
#   2. Agent — LLM decides which tool(s) to call
#   3. Tool execution — Python does the actual work
#   4. Answer — LLM formats the tool result into a natural response
#
# Tools available (tools.py):
#   search_files             — semantic file search
#   count_files              — exact count via db.get()
#   find_folder              — locate project/directory
#   find_folders_with_subfolder — folders containing a named subdir
#   most_recent_items        — most recently modified files or folders
# =============================================================================

import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    OPENROUTER_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SKIP_DIRS_COMMON,
    SKIP_DIRS_WINDOWS,
    SKIP_DIRS_LINUX,
    SKIP_DIRS_MACOS,
)
from tools import make_tools


# ---------------------------------------------------------------------------
# Skip-dir detection
# ---------------------------------------------------------------------------

_ALL_SKIP_DIRS: frozenset[str] = frozenset(
    d.lower() for d in (
        SKIP_DIRS_COMMON | SKIP_DIRS_WINDOWS | SKIP_DIRS_LINUX | SKIP_DIRS_MACOS
    )
)

def _is_skipped_dir_query(q: str) -> bool:
    return q.strip().lower() in _ALL_SKIP_DIRS


# ---------------------------------------------------------------------------
# Greeting detection
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^(hi+|hello|hey|sup|yo|howdy|greetings|what can (you|u) do|how can (you|u) help|help me|what (are|r) you)",
    re.IGNORECASE,
)

_GREETING_SYSTEM = """\
You are feXtch, a local filesystem search assistant.
The user said something casual or asked what you can do.
Respond naturally and briefly in 2-3 sentences. Mention that you can:
find files, locate project folders, count files, find folders with specific subfolders,
and show recently modified items. Keep it friendly and short.
"""

def _is_greeting(q: str) -> bool:
    return bool(_GREETING_RE.match(q.strip()))

def _greeting_response(q: str, llm) -> str:
    result = llm.invoke([
        SystemMessage(content=_GREETING_SYSTEM),
        HumanMessage(content=q),
    ])
    return result.content.strip()


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM = """\
You are feXtch, a local filesystem search assistant.

RULE 1 — ALWAYS call a tool first. Never answer from memory or skip tool calls.

TOOL SELECTION:
- NEVER assume a drive unless user explicitly says "D drive" or "C drive".
- "how many", "count", "total", "how much" → count_files
  (set count_folders=True when asking about folders specifically)
  count_files returns the file list too — do NOT call search_files afterwards.
- "most recent", "latest modified", "recently changed", "made in <month>", "created in <year>", "from <month> <year>" → most_recent_items
  Extract date range: "september 2025" → modified_after=20250901, modified_before=20250930
  "2024" → modified_after=20240101, modified_before=20241231
  "last month" → estimate based on today. "folders made in X" → item_type="folder"
- "where is X", "find X folder", bare project/folder name, "files in X folder", "files inside X" → find_folder
- "folders containing X subfolder" → find_folders_with_subfolder
- everything else, file searches, listing files → search_files

RULE 2 — OUTPUT FORMAT (critical):
After getting a tool result, output it EXACTLY as returned by the tool.
Do NOT paraphrase, summarise, or recount.
Do NOT write "there are X files" — just output what the tool returned.
You may add ONE line at the very top like "Here are the results:" but nothing more.
Preserve all 📁 icons, • bullets, path/ext/size/modified fields exactly.
"""


# ---------------------------------------------------------------------------
# Condense — only rewrites genuine follow-ups
# ---------------------------------------------------------------------------

CONDENSE_SYSTEM = """\
You are a query rewriter for a local filesystem search engine.

Return the question EXACTLY as-is UNLESS it contains a reference word ("it", "those",
"that folder", "the ones") that only makes sense with prior context.

STRICT RULES:
- NEVER add drive letters the user did not mention.
- NEVER add folder names, file types, or any context not in the original.
- "csv files bigger than 5 mb" is already complete — return it unchanged.
- Output ONLY the query text. Nothing else.
"""

def _condense(question: str, history: list[tuple[str, str]], llm) -> str:
    if not history:
        return question
    hist_text = "\n".join(
        f"{'User' if r == 'human' else 'feXtch'}: {c}"
        for r, c in history
    ) or "(none)"
    result = llm.invoke([
        SystemMessage(content=CONDENSE_SYSTEM),
        HumanMessage(content=f"History:\n{hist_text}\n\nFollow-up: {question}\nOutput:"),
    ])
    standalone = result.content.strip().strip('"').strip("'")
    # safety: if condensed is way longer than original, it added context — revert
    if len(standalone) > len(question) * 1.5:
        return question
    return standalone if standalone else question


# ---------------------------------------------------------------------------
# Agent execution loop
# ---------------------------------------------------------------------------


def _run_agent(question: str, llm_with_tools, tools_by_name: dict, history: list) -> str:
    """
    Run the tool-calling agent.
    Tracks tool outputs so if the LLM summarises instead of pasting,
    we return the raw tool output directly.
    """
    messages = [SystemMessage(content=AGENT_SYSTEM)]
    for role, content in history[-6:]:
        messages.append(HumanMessage(content=content) if role == "human" else AIMessage(content=content))
    messages.append(HumanMessage(content=question))

    last_tool_outputs: list[str] = []  # track what tools actually returned

    for iteration in range(6):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            if iteration == 0:
                # LLM skipped tools entirely — nudge once
                messages.append(HumanMessage(
                    content="You must call a tool first. Call the appropriate tool now."
                ))
                continue

            answer = response.content.strip()

            # If the LLM returned a short summary (no bullets, no 📁) but we have
            # rich tool output, return the tool output directly — the LLM paraphrased.
            if last_tool_outputs:
                combined = "\n\n".join(last_tool_outputs)
                has_structure = ("•" in combined or "📁" in combined or
                                 "path" in combined or "\n" in combined[:200])
                llm_is_sparse  = len(answer) < 120 and "•" not in answer and "📁" not in answer
                if has_structure and llm_is_sparse:
                    return combined

            return answer

        for tc in response.tool_calls:
            tool_fn = tools_by_name.get(tc["name"])
            if tool_fn is None:
                result = f"(unknown tool: {tc['name']})"
            else:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"(tool error: {e})"
            last_tool_outputs.append(str(result))
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    messages.append(HumanMessage(content="Give your final answer based on the tool results above."))
    final = llm_with_tools.invoke(messages).content.strip()
    # same fallback for exhausted loop
    if last_tool_outputs:
        combined = "\n\n".join(last_tool_outputs)
        if len(final) < 120 and "•" not in final and "📁" not in final:
            if "•" in combined or "📁" in combined or "path" in combined:
                return combined
    return final

# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def build_pipeline(llm_model: str | None = None):
    llm_model = llm_model or os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )

    llm = init_chat_model(
        model=llm_model,
        model_provider="openai",
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    tool_list   = make_tools(db, db, llm=llm)   # llm powers SelfQueryRetriever in search_files
    tools_by_name = {t.name: t for t in tool_list}
    llm_with_tools = llm.bind_tools(tool_list)

    return llm, llm_with_tools, tools_by_name


# ---------------------------------------------------------------------------
# Interactive chatbot
# ---------------------------------------------------------------------------

def run_chat():
    print("\n" + "=" * 56)
    print("  feXtch — local semantic file search")
    print("  type your query.  'quit' to exit.")
    print("=" * 56 + "\n")

    llm, llm_with_tools, tools_by_name = build_pipeline()
    history: list[tuple[str, str]] = []

    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nbye.")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q", "bye"}:
            print("bye.")
            break

        if _is_greeting(question):
            answer = _greeting_response(question, llm)
            print(f"\nfeXtch > {answer}\n")
            history.append(("human", question))
            history.append(("assistant", answer))
            continue

        # Deleted files query — feXtch can't track deletions in real time
        if any(w in question.lower().split() for w in
               ('deleted', 'removed', 'trash', 'recycle', 'bin', 'gone', 'missing')):
            del_msg = 'feXtch cannot track deleted files -- only indexes existing files. '\
                      'Run syncer.py to find what was removed since last sync.'
            print(f'\nfeXtch > {del_msg}\n')
            history.append(('human', question))
            history.append(('assistant', del_msg))
            continue


        # Deleted files query
        _del_words = {'deleted','removed','trash','recycle','bin','gone','missing'}
        if any(w in question.lower().split() for w in _del_words):
            answer = (
                "feXtch cannot track deleted files — it only indexes files that exist on disk.\n"
                "To find what was deleted since your last sync, run syncer.py:\n"
                "  python rag\\syncer.py\n"
                "Syncer compares the current filesystem against the index and reports removed entries."
            )
            print(f"\nfeXtch > {answer}\n")
            history.append(("human", question))
            history.append(("assistant", answer))
            continue

        _del_words = {'deleted','removed','trash','recycle','bin','gone','missing'}
        if any(w in question.lower().split() for w in _del_words):
            del_msg = (
                'feXtch cannot track deleted files '
                '-- it only indexes files that exist on disk.\n'
                'To find what was deleted since your last sync, run:\n'
                '  python rag\\syncer.py\n'
                'Syncer compares filesystem against index and reports removed entries.'
            )
            print(f'\nfeXtch > {del_msg}\n')
            history.append(('human', question))
            history.append(('assistant', del_msg))
            continue

        if _is_skipped_dir_query(question):
            answer = (
                f"feXtch doesn't index '{question.strip()}' directories — "
                "they contain package manager or build artefacts, not user files. "
                "To include them, remove the name from SKIP_DIRS_COMMON in config.py "
                "and re-run the indexer."
            )
            print(f"\nfeXtch > {answer}\n")
            history.append(("human", question))
            history.append(("assistant", answer))
            continue

        standalone = _condense(question, history, llm)
        if standalone != question:
            print(f"\n[condensed → \"{standalone}\"]\n")

        answer = _run_agent(standalone, llm_with_tools, tools_by_name, history)
        print(f"\nfeXtch > {answer}\n")

        history.append(("human", question))
        history.append(("assistant", answer))


if __name__ == "__main__":
    run_chat()