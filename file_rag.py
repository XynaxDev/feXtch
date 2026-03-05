from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.chains.query_constructor.base import (
    load_query_constructor_runnable,
)
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from datetime import datetime
import string

load_dotenv()

# LLM
llm = init_chat_model(
    model=os.getenv("LLM_MODEL"),
    model_provider="openai",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)

# Embedding Model
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Detect all Drives on your local system
def get_drives():
    drives = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            drives.append(drive)
    return drives


drives = get_drives()
print(drives)

# multi-drive smart file system scanner

SKIP_DIRS = {
    "AppData",
    "ProgramData",
    "Windows",
    "$Recycle.Bin",
    "System Volume Information",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".git",
    "dist",
    "build",
    "target",
    ".idea",
    ".vscode",
}

SKIP_EXTENSIONS = {
    "sys",
    "tmp",
    "log",
    "dat",
    "cache",
    "iso",
    "bin",
}

MAX_FILES_C_DRIVE = 5000


def scan_drives(drives):

    docs = []
    drive_stats = {}

    for drive in drives:
        print(f"\nScanning drive: {drive}")

        file_count = 0
        limit_reached = False

        for root, dirs, files in os.walk(drive, topdown=True):
            if limit_reached:
                break

            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for f in files:
                path = os.path.join(root, f)

                try:
                    stat = os.stat(path)
                    ext = os.path.splitext(f)[1].replace(".", "").lower()
                    # skip hidden files
                    if f.startswith("."):
                        continue

                    # skip junk extensions
                    if ext in SKIP_EXTENSIONS:
                        continue

                    folder_name = os.path.basename(root)
                    # for better retrieval and better semantic meaning
                    page_text = f"{f} {ext} file in folder {folder_name}"

                    doc = Document(
                        page_content=page_text,
                        metadata={
                            "filename": f,
                            "path": path,
                            "folder": folder_name,
                            "extension": ext,
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "created_at": datetime.fromtimestamp(
                                stat.st_ctime
                            ).strftime("%Y-%m-%d"),
                            "modified_at": datetime.fromtimestamp(
                                stat.st_mtime
                            ).strftime("%Y-%m-%d"),
                            "drive": drive,
                        },
                    )

                    docs.append(doc)
                    file_count += 1

                    if drive.startswith("C") and file_count >= MAX_FILES_C_DRIVE:
                        print("C drive limit reached")
                        limit_reached = True
                        break

                except (PermissionError, FileNotFoundError, OSError):
                    continue

        drive_stats[drive] = file_count
        print(f"Indexed {file_count} files from {drive}")

    return docs, drive_stats

# Run the scanner
drives = get_drives()
docs, stats = scan_drives(drives)

print(len(docs))
print(stats)

# See 1st 10 docs
for d in docs[:10]:
    print(f"Page Content: {d.page_content}")
    print(f"Metadata: {d.metadata}")
    print("-" * 60)


