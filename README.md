# feXtch

![Python](https://img.shields.io/badge/python-3.11-blue?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/langchain-framework-black?style=flat)
![ChromaDB](https://img.shields.io/badge/vector--db-chromadb-purple?style=flat)
![Ollama](https://img.shields.io/badge/embeddings-ollama-green?style=flat)
![License](https://img.shields.io/badge/license-MIT-orange?style=flat)

**feXtch** (Fast EXtended file feTCH engine) is a local AI-powered semantic file search engine that lets you search your filesystem using natural language instead of exact filenames.

It indexes file names and metadata, generates embeddings, and retrieves relevant files using vector similarity search.

Everything runs **locally**, keeping your files private.

## Features

• Semantic file search using embeddings
• Natural language queries for files
• Multi-drive indexing (C, D, E, etc.)
• Metadata-aware search (extension, size, dates)
• Incremental indexing when files change
• Fully local AI pipeline

Example queries:

```
python notebooks about deep learning
pdf files created after 2024
files related to cnn training
large datasets in research folder
```

## Tech Stack

* Python 3.11
* LangChain
* Ollama (local embeddings)
* ChromaDB (vector database)
* OpenRouter (LLM support)
* Watchdog (filesystem monitoring)

## Installation

Clone the repo:

```
git clone https://github.com/XynaxDev/feXtch.git
cd feXtch
```

Create virtual environment:

```
python -m venv venv
```

Activate environment:

Windows

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

## Install Embedding Model

Install Ollama from:

https://ollama.com

Then pull the embedding model:

```
ollama pull nomic-embed-text
```

## Run the Indexer

```
python indexer.py
```

This scans selected directories and stores embeddings in a local Chroma database.

## Example Indexed Document

```
page_content:
"train_model.py py file in folder DeepLearning"

metadata:
{
  filename: "train_model.py",
  path: "D:\AI\DeepLearning\train_model.py",
  extension: "py",
  size_mb: 0.3
}
```

## Roadmap

* File content indexing (PDF, code, notebooks)
* Knowledge graph between files
* Desktop UI (Tauri / PySide)
* Agentic search

## License
MIT License
