import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredEmailLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# --- CONFIGURATION ---

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda"


@dataclass
class DataSource:
    name: str
    path: str
    type: Literal["text", "code", "email"]
    glob: str = "**/*"
    priority: str = "Low"
    language: Literal["python", "bash", None] = None


def load_config(config_files: list[str]) -> list[DataSource]:
    """Load data sources from YAML/JSON config files."""
    sources = []
    for config_file in config_files:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"Config file not found: {config_file}")
            continue
        print(f"Loading config: {config_file}")
        with open(config_path) as f:
            if (
                config_path.suffix.lower() == ".yaml"
                or config_path.suffix.lower() == ".yml"
            ):
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                print(f"Unsupported file format: {config_path.suffix}")
                continue
        # Parse sources list
        sources_list = data.get("sources", [])
        for source_data in sources_list:
            source = DataSource(**source_data)
            sources.append(source)

    return sources


def compute_hash(paths: list[str]) -> str:
    """Compute SHA256 hash of all data paths."""
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(path.encode("utf-8"))
    return hasher.hexdigest()[:12]


def get_output_path(output_name: str = None, data_paths: list[str] = None) -> str:
    """
    Get output index path.
    If output_name is provided, use it.
    Otherwise, hash all data paths.
    """
    if output_name:
        return output_name
    if not data_paths:
        return "faiss_index_unified"
    hash_value = compute_hash(data_paths)
    return f"faiss_index_{hash_value}"


def load_text(source: DataSource) -> list[Document]:
    """Loads standard text files (MD, DOCX, TXT, SH)."""
    print(f"Loading Text: {source.name}...")
    loader = DirectoryLoader(
        source.path,
        glob=source.glob,
        loader_cls=UnstructuredMarkdownLoader if ".md" in source.glob else None,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def load_email(source: DataSource) -> list[Document]:
    """Loads .eml files using UnstructuredEmailLoader."""
    print(f"Loading Emails: {source.name}...")
    loader = DirectoryLoader(
        source.path,
        glob=source.glob,
        loader_cls=UnstructuredEmailLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def load_code(source: DataSource) -> list[Document]:
    """Loads Code using LanguageParser for smarter splitting."""
    print(f"Loading Code: {source.name}...")
    loader = GenericLoader.from_filesystem(
        source.path,
        glob=source.glob,
        parser=LanguageParser(language=source.language, parser_threshold=30),
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    return splitter.split_documents(docs)


def build_index(sources: list[DataSource], output_path: str):
    """Build and save FAISS index from data sources."""
    all_chunks = []
    for source in sources:
        if not os.path.exists(source.path):
            print(f"Skipping {source.name}: Path not found ({source.path})")
            continue
        if source.type == "text":
            chunks = load_text(source)
        elif source.type == "code":
            chunks = load_code(source)
        elif source.type == "email":
            chunks = load_email(source)
        else:
            print(f"Unknown type: {source.type}")
            continue
        print(f"Tagging {len(chunks)} chunks...")
        for doc in chunks:
            doc.metadata["source_name"] = source.name
            doc.metadata["priority"] = source.priority
            if "source" in doc.metadata:
                doc.metadata["filename"] = os.path.basename(doc.metadata["source"])
        all_chunks.extend(chunks)
    if not all_chunks:
        print("No documents loaded. Exiting.")
        return
    print(f"\nTotal chunks to index: {len(all_chunks)}")
    print(f"Initializing Embeddings ({DEVICE})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": DEVICE}
    )
    print("Building FAISS Index...")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(output_path)
    print(f"Index saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from data sources"
    )
    parser.add_argument(
        "config_files",
        nargs="+",
        help="YAML or JSON config files defining data sources",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output index path (default: hash of data paths)",
    )

    args = parser.parse_args()
    sources = load_config(args.config_files)
    if not sources:
        print("No sources loaded from config files.")
        exit(1)
    data_paths = [s.path for s in sources]
    output_path = get_output_path(args.output, data_paths)
    print(f"Using output path: {output_path}\n")
    start = time.time()
    build_index(sources, output_path)
    print(f"Total time: {time.time() - start:.2f}s")
