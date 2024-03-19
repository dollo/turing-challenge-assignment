import os

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter


def get_db(path: str):
    """
    Create ChromaDB instance and upload documents in a path.
    """
    # Load docs
    documents = PyPDFDirectoryLoader(path).load()

    # Chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Load into Chroma using embeddings
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    return db
