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
from tqdm import tqdm

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda"

@dataclass
class DataSource:
    name: str
    path: str
    type: Literal["text", "code", "email"]
    glob: str = "**/*"
    priority: str = "Low"
    language: str = None

def load_config(config_files: list[str]) -> list[DataSource]:
    sources = []
    for config_file in config_files:
        path = Path(config_file)
        if not path.exists():
            continue
        with open(path) as f:
            data = yaml.safe_load(f) if path.suffix.lower() in [".yaml", ".yml"] else json.load(f)
        for s_data in data.get("sources", []):
            valid_data = {k: v for k, v in s_data.items() if k in DataSource.__dataclass_fields__}
            sources.append(DataSource(**valid_data))
    return sources

def compute_hash(paths: list[str]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(path.encode("utf-8"))
    return hasher.hexdigest()[:12]

def get_output_path(output_name: str = None, data_paths: list[str] = None) -> str:
    if output_name:
        return output_name
    return f"faiss_index_{compute_hash(data_paths)}" if data_paths else "faiss_index_unified"

def load_text(source: DataSource) -> list[Document]:
    print(f"Loading Text: {source.name}")
    loader_cls = UnstructuredMarkdownLoader if ".md" in source.glob.lower() else None
    kwargs = {"loader_cls": loader_cls} if loader_cls else {}
    loader = DirectoryLoader(
        source.path,
        glob=source.glob,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True,
        **kwargs,
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def load_code(source: DataSource) -> list[Document]:
    print(f"Loading Code: {source.name}")
    supported_parsers = ["python", "cpp", "js", "c", "ts", "html", "markdown"]
    safe_lang = source.language if source.language in supported_parsers else None
    loader = GenericLoader.from_filesystem(
        source.path,
        glob=source.glob,
        parser=LanguageParser(language=safe_lang, parser_threshold=30),
    )
    blobs = list(loader.blob_loader.yield_blobs())
    docs = []
    if blobs:
        # Changed: Removed leave=False and added mininterval to force rendering
        for blob in tqdm(blobs, desc=f"  Parsing {source.name}", unit="file", mininterval=0.1):
            try:
                docs.extend(loader.blob_parser.lazy_parse(blob))
            except Exception:
                continue
    print(f"  Found {len(docs)} files")
    lang_map = {"python": Language.PYTHON, "cpp": Language.CPP, "js": Language.JS}
    target_enum = lang_map.get(source.language.lower() if source.language else None)
    splitter = (
        RecursiveCharacterTextSplitter.from_language(language=target_enum, chunk_size=2000, chunk_overlap=200)
        if target_enum else RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    )
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks")
    return chunks

def load_email(source: DataSource) -> list[Document]:
    print(f"Loading Emails: {source.name}")
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

def build_index(sources: list[DataSource], output_path: str):
    all_chunks = []
    for source in sources:
        if not os.path.exists(source.path):
            print(f"Path not found: {source.path}")
            continue
        if source.type == "text":
            chunks = load_text(source)
        elif source.type == "code":
            chunks = load_code(source)
        elif source.type == "email":
            chunks = load_email(source)
        else:
            continue
        for doc in chunks:
            doc.metadata.update({"source_name": source.name, "priority": source.priority})
            if "source" in doc.metadata:
                doc.metadata["filename"] = os.path.basename(doc.metadata["source"])
        all_chunks.extend(chunks)
    if not all_chunks:
        print("No chunks to index.")
        return
    print(f"\nIndexing {len(all_chunks)} total chunks...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": DEVICE})
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(output_path)
    print(f"Index saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_files", nargs="+")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()
    sources = load_config(args.config_files)
    if sources:
        out = get_output_path(args.output, [s.path for s in sources])
        start = time.time()
        build_index(sources, out)
        print(f"Finished in {time.time() - start:.2f}s")