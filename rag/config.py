# =============================================================================
# feXtch — Fast EXtended file feTCH engine
# config.py  |  Central configuration
#
# Author  : Akash Kumar
# Email   : akashkumar.cs27@gmail.com
# GitHub  : github.com/akashkumar
# Project : feXtch — local AI semantic file search
# =============================================================================

from pathlib import Path

# ---------------------------------------------------------------------------
# Drive / directory scanning
# ---------------------------------------------------------------------------

CUSTOM_SCAN_ROOTS: list[str] | None = None

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

CHROMA_DIR: Path = Path("./data/chroma_db")
CHROMA_COLLECTION: str = "fextch-files"

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = "http://localhost:11434"
EMBED_MODEL: str = "nomic-embed-text:v1.5"
EMBED_BATCH_SIZE: int = 512
EMBED_PARALLEL_BATCHES: int = 8

# ---------------------------------------------------------------------------
# Two-phase indexing
# ---------------------------------------------------------------------------

PHASE1_QUICK_INDEX: int = 5_000

# ---------------------------------------------------------------------------
# LLM (OpenRouter)
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_TEMPERATURE: float = 0.2
LLM_MAX_TOKENS: int = 1_024
RETRIEVER_TOP_K: int = 20          # max results per semantic search
RESULTS_SHOW:    int = 10          # files/folders shown per response
FOLDER_SHOW:     int = 10          # diverse folders shown in summaries
APPROX_MARGIN:   float = 0.25      # ±25% range for approx size queries

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

SCORE_WEIGHTS: dict = {
    "extension": 2.0,
    "size":      0.5,
    "recency":   0.8,
    "depth":     1.2,
}

EXTENSION_PRIORITY: dict[str, float] = {
    "ipynb": 1.0, "py": 1.0, "java": 1.0, "cpp": 1.0, "c": 1.0,
    "h": 0.9,  "cs": 1.0, "go": 1.0, "rs": 1.0, "kt": 0.9,
    "js": 0.9, "ts": 0.9, "jsx": 0.9, "tsx": 0.9, "vue": 0.9,
    "rb": 0.9, "php": 0.85, "swift": 0.9, "r": 0.85,
    "pdf": 0.95, "docx": 0.9, "doc": 0.85, "pptx": 0.85, "ppt": 0.8,
    "xlsx": 0.85, "xls": 0.8, "md": 0.95, "txt": 0.8,
    "csv": 0.85, "json": 0.8, "yaml": 0.75, "yml": 0.75,
    "xml": 0.7,  "html": 0.7, "css": 0.65, "sql": 0.8,
    "png": 0.2, "jpg": 0.2, "jpeg": 0.2, "gif": 0.15, "svg": 0.3,
    "mp4": 0.1, "mkv": 0.1, "avi": 0.1, "mov": 0.1, "mp3": 0.1,
    "zip": 0.15, "tar": 0.15, "gz": 0.15, "rar": 0.15,
}

DEFAULT_EXTENSION_PRIORITY: float = 0.4

# ---------------------------------------------------------------------------
# Skip rules — common across all OSes
# ---------------------------------------------------------------------------

SKIP_DIRS_DEPENDENCIES: set[str] = {
    "node_modules", "bower_components",
    "venv", ".venv", "env", "virtualenv",
    "__pycache__", "site-packages", ".tox",
}

SKIP_DIRS_BUILD: set[str] = {
    "dist", "build", "target", "out", "obj", "release", "debug",
}

SKIP_DIRS_VCS_AND_TOOLS: set[str] = {
    ".git", ".svn", ".hg", ".bzr",
    ".idea", ".vscode", ".eclipse",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".hypothesis",
    "tmp", "temp", ".tmp",
}

SKIP_DIRS_COMMON: set[str] = (
    SKIP_DIRS_DEPENDENCIES
    | SKIP_DIRS_BUILD
    | SKIP_DIRS_VCS_AND_TOOLS
)

# ---------------------------------------------------------------------------
# Windows-specific skip rules
# ---------------------------------------------------------------------------

SKIP_DIRS_WINDOWS: set[str] = {
    "Windows", "ProgramData", "$Recycle.Bin", "System Volume Information",
    "Program Files", "Program Files (x86)",
    "AppData", "Local", "LocalLow", "Roaming",
    "Temp", "Temporary Internet Files", "INetCache", "INetCookies",
    "Packages", "Microsoft", "MicrosoftEdge", "Google", "Mozilla",
    "PrintHood", "NetHood", "Recent", "SendTo", "Templates",
}

SKIP_PATH_FRAGMENTS_WINDOWS: set[str] = {
    "\\appdata\\", "\\windows\\", "\\programdata\\",
    "\\program files\\", "\\program files (x86)\\",
    "\\system32\\", "\\syswow64\\", "\\winsxs\\",
    "\\temp\\", "\\powershell\\", "\\inetcache\\",
    "\\microsoft\\windows\\",
}

# ---------------------------------------------------------------------------
# Linux-specific skip rules
# ---------------------------------------------------------------------------

SKIP_DIRS_LINUX: set[str] = {
    "proc", "sys", "dev", "run", "snap",
    "boot", "sbin", "lib", "lib64", "usr", "bin", "etc", "var",
    ".cache", ".config", ".dbus", ".local", ".gnome", ".kde",
    ".mozilla", ".thunderbird",
}

SKIP_PATH_FRAGMENTS_LINUX: set[str] = {
    "/proc/", "/sys/", "/dev/", "/run/",
    "/.cache/", "/.config/", "/.local/share/", "/snap/",
}

# ---------------------------------------------------------------------------
# macOS-specific skip rules
# ---------------------------------------------------------------------------

SKIP_DIRS_MACOS: set[str] = {
    "System", "private", "cores",
    "usr", "bin", "sbin", "etc", "dev",
    "Library", "Caches", "Preferences", "Application Support",
    "Containers", "Logs", "Mail", "Safari",
    ".Spotlight-V100", ".fseventsd", ".Trash",
}

SKIP_PATH_FRAGMENTS_MACOS: set[str] = {
    "/system/", "/private/var/", "/private/tmp/",
    "/library/caches/", "/library/logs/",
    "/library/containers/", "/library/application support/",
    "/.trash/",
}

# ---------------------------------------------------------------------------
# Extension blocklist
# ---------------------------------------------------------------------------

SKIP_EXTENSIONS: set[str] = {
    # OS / system binaries — not user content
    "sys", "dll", "so", "dylib", "drv", "ocx", "mui",
    # Windows executables and installers
    # exe is critical: catches flask.exe, python.exe etc. inside venv Scripts/
    "exe", "msi", "cab", "msp", "msu",
    # PowerShell
    "ps1", "psm1", "psd1",
    # temp / cache / logs
    "tmp", "temp", "log", "cache", "bak", "old",
    # compiled bytecode
    "pyc", "pyo", "pyd", "class",
    # disk images
    "iso", "img", "bin", "dat",
    # lock / marker files
    "lock", "pid",
    # Windows shortcut / metadata
    "lnk", "url", "ini", "inf",
    # macOS metadata
    "ds_store",
}

SKIP_PREFIX: str = "."