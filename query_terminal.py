import json
import os
from enum import StrEnum, auto

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

CONFIG = None


class ConfigKeys(StrEnum):
    # auto() creates the value "index_path" from the name
    INDEX_PATH = auto()
    EMBEDDING_MODEL = auto()
    EMBEDDING_DEVICE = auto()
    MODEL_PATH = auto()
    RETRIEVER_K = auto()
    SYSTEM_PROMPT = auto()


def validate_config(loaded_json: dict) -> bool:
    for key in loaded_json:
        if key not in ConfigKeys._value2member_map_:
            print(f"Invalid key detected: {key}")
    required_keys = {ConfigKeys.INDEX_PATH, ConfigKeys.MODEL_PATH}
    current_keys = set(loaded_json.keys())
    missing = required_keys - current_keys
    if missing:
        print(f"Missing keys: {missing}")
        return False
    return True


def load_rag_chain(cfg) -> RunnableParallel:
    index_path = cfg.get(ConfigKeys.index_path)
    embedding_model = cfg.get(ConfigKeys.embedding_model)
    embedding_device = cfg.get(ConfigKeys.embedding_device)
    model_path = cfg.get(ConfigKeys.model_path)
    retriever_k = cfg.get(ConfigKeys.retriever_k)
    system_prompt = cfg.get(ConfigKeys.system_prompt)

    print("\n--- Initializing RAG Pipeline with Citations ---")
    print(f"Loading Index from {index_path}...")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": embedding_device},
    )

    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    print(f"Loading LLM: {os.path.basename(model_path)}...")
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=cfg.get("llm_n_gpu_layers", -1),
        n_ctx=cfg.get("llm_n_ctx", 8192),
        n_batch=cfg.get("llm_n_batch", 512),
        verbose=False,
        callback_manager=callback_manager,
        temperature=0.1,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    rag_chain = (
        RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
        .assign(formatted_context=lambda x: format_docs_with_source(x["context"]))
        .assign(answer=prompt | llm | StrOutputParser())
        .pick(["answer", "context"])
    )
    print("✅ Pipeline Ready.")
    return rag_chain


def load_config(path: str):
    global CONFIG
    with open(path) as f:
        CONFIG = json.load(f)


def format_docs_with_source(docs: list[Document]) -> str:
    """
    Formats documents for the LLM, injecting the source tag explicitly.
    """
    formatted_chunks = []
    for doc in docs:
        source_name = doc.metadata.get("source_name", "Unknown")
        # Use filename if available, otherwise source path
        filename = doc.metadata.get(
            "filename", doc.metadata.get("source", "Unknown File")
        )
        content = doc.page_content
        # Explicit XML-style tags for the model
        chunk_str = f"Source: {source_name} | File: {filename}\nContent:\n{content}\n"
        formatted_chunks.append(chunk_str)
    return "\n---\n".join(formatted_chunks)


def print_sources(docs: list[Document]):
    """Pretty prints the sources used for the answer."""
    print("\n" + "=" * 60)
    print(f"SOURCES USED ({len(docs)})")
    print("=" * 60)

    # Group by Source Type for cleaner output
    sources_by_type = {}
    for doc in docs:
        s_type = doc.metadata.get("source_name", "General")
        if s_type not in sources_by_type:
            sources_by_type[s_type] = []
        sources_by_type[s_type].append(doc)

    for s_type, doc_list in sources_by_type.items():
        print(f"\n🔹 [{s_type}]")
        seen_files = set()
        for doc in doc_list:
            # Try to find the cleanest display name
            # If 'filename' was set in the builder script, use it
            # Otherwise fallback to the full path 'source'
            file_display = doc.metadata.get(
                "filename", doc.metadata.get("source", "Unknown")
            )
            # Avoid printing duplicate filenames if multiple chunks came from same file
            if file_display in seen_files:
                continue
            seen_files.add(file_display)
            print(f"   📄 {file_display}")
            if "subject" in doc.metadata:
                print(f"      Subject: {doc.metadata['subject']}")
    print("=" * 60 + "\n")


def main():
    chain = load_rag_chain()
    print("\nType your query (or 'exit')...")
    while True:
        try:
            query = input("\nQuery: ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query.strip():
                continue
            print("\nAnswer: ", end="", flush=True)
            result = chain.invoke(query)
            print_sources(result["context"])
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
