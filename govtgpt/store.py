"""Step 3: embeddings + Qdrant (local mode, data saved in ./qdrant_db)."""
from __future__ import annotations

import os
import time
import uuid

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# EMBED_BACKEND=gemini (default) or bge. BGE gets its own DB + query cache so the
# two indexes can be compared side by side.
BACKEND = os.getenv("EMBED_BACKEND", "gemini").lower()

if BACKEND == "bge":
    EMBED_MODEL = os.getenv("BGE_MODEL", "BAAI/bge-m3")      # multilingual, 1024-dim
    DIM = {"BAAI/bge-m3": 1024, "BAAI/bge-small-en-v1.5": 384,
           "BAAI/bge-base-en-v1.5": 768, "BAAI/bge-large-en-v1.5": 1024}[EMBED_MODEL]
    _tag = "" if EMBED_MODEL == "BAAI/bge-m3" else "_" + EMBED_MODEL.split("/")[1]
    DB_PATH = f"qdrant_db_bge{_tag}"
    QUERY_CACHE = f"artifacts/query_embeddings_bge{_tag}.json"
else:
    EMBED_MODEL = "models/gemini-embedding-2"
    DIM = 3072
    DB_PATH = "qdrant_db"
    QUERY_CACHE = "artifacts/query_embeddings.json"
COLLECTION = "govtgpt"


def get_embeddings():
    if BACKEND == "bge":
        from langchain_huggingface import HuggingFaceEmbeddings
        # en-v1.5 models want an instruction prefix on queries; bge-m3 does not.
        qi = ("Represent this sentence for searching relevant passages: "
              if "en-v1.5" in EMBED_MODEL else "")
        return HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
            query_encode_kwargs={"normalize_embeddings": True, "prompt": qi} if qi
            else {"normalize_embeddings": True})
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def get_client() -> QdrantClient:
    return QdrantClient(path=DB_PATH)      # local mode: no server needed


def build_index(docs: list[Document], batch: int = 20) -> QdrantVectorStore:
    """Embed all chunks and store them. Re-running recreates the collection."""
    client = get_client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )
    store = QdrantVectorStore(client=client, collection_name=COLLECTION,
                              embedding=get_embeddings())
    # Stable id per chunk: same chunk -> same id, so a retry or re-run overwrites
    # instead of creating duplicates.
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL,
                          f"{d.metadata['short']}|{d.metadata['section']}|"
                          f"{d.metadata['chunk_index']}|{d.page_content}"))
           for d in docs]
    assert len(set(ids)) == len(ids), "duplicate chunk ids"
    for i in range(0, len(docs), batch):
        for attempt in range(5):           # simple retry for rate limits
            try:
                store.add_documents(docs[i:i + batch], ids=ids[i:i + batch])
                break
            except Exception as e:
                print(f"  retry {attempt + 1}: {str(e)[:160]}")
                time.sleep(30 * (attempt + 1))
        else:
            raise RuntimeError("embedding failed repeatedly")
        print(f"  embedded {min(i + batch, len(docs))}/{len(docs)}")
    return store


def load_store() -> QdrantVectorStore:
    return QdrantVectorStore(client=get_client(), collection_name=COLLECTION,
                             embedding=get_embeddings())
