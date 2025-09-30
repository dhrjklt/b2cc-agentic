#!/usr/bin/env python
"""
Ingest local PDFs/TXTs -> split -> embed -> store in Chroma
- Semantic-aware chunking with overlap
- Adds metadata (filename, page)
- Uses stronger embeddings (all-mpnet-base-v2 by default)
- Always rebuilds the Chroma DB (no duplicates)
"""

import os
import shutil
import typer
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

app = typer.Typer(add_completion=False)


def load_documents(data_dir: str):
    """Load PDFs and TXTs into LangChain documents with metadata."""
    docs = []
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["filename"] = file
            docs.extend(file_docs)

        elif file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["filename"] = file
            docs.extend(file_docs)

    return docs


def build_chroma(
    data_dir="./docs",
    db_dir="./chroma_db",
    model="sentence-transformers/all-mpnet-base-v2",
):
    docs = load_documents(data_dir)

    # Semantic-aware chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Stronger embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model)

    # Always rebuild DB: delete existing
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )

    return vectordb, len(docs), len(chunks)


@app.command()
def ingest(
    data_dir: str = typer.Option("./docs", help="Folder with PDF/TXT files"),
    db_dir: str = typer.Option("./chroma_db", help="Directory to store Chroma DB"),
    model: str = typer.Option("sentence-transformers/all-mpnet-base-v2", help="Embedding model"),
):
    """
    Run the ingestion pipeline and save to Chroma DB.
    """
    vectordb, n_docs, n_chunks = build_chroma(data_dir, db_dir, model)
    typer.echo(f"📂 Loaded {n_docs} docs, split into {n_chunks} chunks")
    typer.echo(f"✅ Chroma DB rebuilt at {db_dir}")


if __name__ == "__main__":
    app()