import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Config Loader ---


def load_config(env_path: str = '.env') -> Dict[str, str]:
    """Load configuration from .env file and environment variables."""
    load_dotenv(env_path, override=True)
    try:
        return {
            'DATA_DIR': os.getenv('DATA_DIR', 'data'),
            'COLLECTION_NAME': os.getenv('COLLECTION_NAME', 'documents'),
            'EMBEDDING_DIM': int(os.getenv('EMBEDDING_DIM', 384)),
            'CHUNK_SIZE': int(os.getenv('CHUNK_SIZE', 500)),
            'CHUNK_OVERLAP': int(os.getenv('CHUNK_OVERLAP', 50)),
            'MODEL_NAME': os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2'),
            'BATCH_SIZE': int(os.getenv('BATCH_SIZE', 64)),
            'QDRANT_HOST': os.getenv('QDRANT_HOST', 'localhost'),
            'QDRANT_PORT': int(os.getenv('QDRANT_PORT', 6333)),
        }
    except Exception as e:
        raise RuntimeError(f"Config error: {e}")

# --- Document Loader ---


def load_documents(data_dir: str) -> List[Tuple[str, str]]:
    """Load all .txt files from data_dir."""
    path = Path(data_dir)
    if not path.exists() or not path.is_dir():
        logging.warning(f"Data directory {data_dir} does not exist.")
        return []
    docs = []
    for file in path.glob('*.txt'):
        try:
            with file.open('r', encoding='utf-8') as f:
                docs.append((file.name, f.read()))
        except Exception as e:
            logging.error(f"Failed to read {file}: {e}")
    return docs

# --- Text Splitter ---


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks. No trailing chunk if too short."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size/overlap")
    chunks = []
    start = 0
    while start + chunk_size <= len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Embedding Model Factory ---


class EmbeddingModel:
    """SentenceTransformer embedding wrapper."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc='Embedding'):
            batch = texts[i:i+batch_size]
            emb = self.model.encode(batch, show_progress_bar=False, device=self.device,
                                    convert_to_numpy=True, normalize_embeddings=True)
            embeddings.append(emb.astype(np.float32))
        return np.vstack(embeddings)

# --- Vector Store Factory (Qdrant) ---


class QdrantStore:
    """Qdrant vector DB wrapper for document chunks."""

    def __init__(self, host: str, port: int, collection_name: str, embedding_dim: int):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = QdrantClient(host=self.host, port=self.port)
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, distance=Distance.COSINE)
            )

    def insert(self, embeddings: np.ndarray, contents: List[str], batch_size: int = 64):
        if len(embeddings) != len(contents):
            raise ValueError(
                "Embeddings and contents must have the same length.")
        points = []
        for idx, (vec, text) in enumerate(zip(embeddings, contents)):
            points.append(PointStruct(
                id=idx, vector=vec.tolist(), payload={"content": text}))
        for i in tqdm(range(0, len(points), batch_size), desc='Qdrant Insert'):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name, points=batch)

# --- Main Ingestion Pipeline ---


def ingest_pipeline(cfg: Optional[Dict[str, str]] = None) -> None:
    """Main ingestion pipeline. Loads config, documents, embeds, and stores."""
    cfg = cfg or load_config()
    logging.info(f"Loaded config: {cfg}")
    docs = load_documents(cfg['DATA_DIR'])
    if not docs:
        logging.warning("No documents found for ingestion.")
        return
    model = EmbeddingModel(cfg['MODEL_NAME'])
    store = QdrantStore(cfg['QDRANT_HOST'], cfg['QDRANT_PORT'],
                        cfg['COLLECTION_NAME'], cfg['EMBEDDING_DIM'])
    for fname, text in docs:
        try:
            chunks = split_text(text, cfg['CHUNK_SIZE'], cfg['CHUNK_OVERLAP'])
        except Exception as e:
            logging.error(f"Failed to split {fname}: {e}")
            continue
        if not chunks:
            logging.warning(f"No chunks produced for {fname}.")
            continue
        embeddings = model.embed(chunks, batch_size=cfg['BATCH_SIZE'])
        store.insert(embeddings, chunks, batch_size=cfg['BATCH_SIZE'])
        logging.info(f"Ingested {len(chunks)} chunks from {fname}")
    logging.info("Ingestion complete.")


def main():
    ingest_pipeline()


if __name__ == "__main__":
    main()
