import os
import sys
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import openai
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
            'COLLECTION_NAME': os.getenv('COLLECTION_NAME', 'documents'),
            'EMBEDDING_DIM': int(os.getenv('EMBEDDING_DIM', 384)),
            'MODEL_NAME': os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2'),
            'QDRANT_HOST': os.getenv('QDRANT_HOST', 'localhost'),
            'QDRANT_PORT': int(os.getenv('QDRANT_PORT', 6333)),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        }
    except Exception as e:
        raise RuntimeError(f"Config error: {e}")

# --- Embedding Model Factory ---


class EmbeddingModel:
    """SentenceTransformer embedding wrapper."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], show_progress_bar=False)
        return np.array(emb, dtype=np.float32)

# --- Vector Store Factory (Qdrant) ---


class QdrantStore:
    """Qdrant vector DB wrapper for query pipeline."""

    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[str]:
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector[0],
            limit=top_k,
        )
        return [hit.payload["content"] for hit in search_result]

# --- LLM Client (OpenAI) ---


class OpenAIClient:
    """OpenAI LLM client for RAG pipeline."""

    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def answer(self, query: str, context: List[str]) -> str:
        prompt = (
            "Answer the following question using only the provided context.\n"
            f"Context:\n{chr(10).join(context)}\n"
            f"Question: {query}\nAnswer:"
        )
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

# --- Query Pipeline ---


def get_query() -> str:
    if len(sys.argv) > 1:
        return ' '.join(sys.argv[1:])
    return input('Enter your query: ')


def query_pipeline(user_query: Optional[str] = None):
    """Main query pipeline. Loads config, embeds query, searches Qdrant, calls OpenAI."""
    cfg = load_config()
    logging.info(f"Loaded config: {cfg}")
    query = user_query or get_query()
    embedder = EmbeddingModel(cfg['MODEL_NAME'])
    store = QdrantStore(cfg['QDRANT_HOST'],
                        cfg['QDRANT_PORT'], cfg['COLLECTION_NAME'])
    llm = OpenAIClient(cfg['OPENAI_API_KEY'])
    query_emb = embedder.embed(query)
    context_chunks = store.search(query_emb, top_k=3)
    if not context_chunks:
        logging.warning('No relevant context found.')
        print('No relevant context found.')
        return
    answer = llm.answer(query, context_chunks)
    print(f"\nAnswer:\n{answer}\n")


def main():
    query_pipeline()


if __name__ == "__main__":
    main()
