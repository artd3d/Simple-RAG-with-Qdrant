# Simple RAG (Retrieval-Augmented Generation) with Qdrant

## Overview
This project implements a simple Retrieval-Augmented Generation (RAG) pipeline using Python, Qdrant vector database, and OpenAI's API. It ingests text documents, splits them into chunks, embeds them, stores them in Qdrant, and allows querying with LLM-augmented answers based on relevant document retrieval.

## Features
- Document ingestion: split, embed, and store text chunks in Qdrant
- Query pipeline: embed query, retrieve top-k relevant chunks, generate answer with OpenAI
- Modular, type-annotated, and testable code
- Dockerized Qdrant setup
- FastAPI HTTP API
- Comprehensive unit tests

## Project Structure
```
python course/
├── data/                # Place your .txt files here
├── ingest.py            # Ingestion pipeline (Qdrant)
├── query.py             # Query pipeline (Qdrant + OpenAI)
├── api.py               # FastAPI server
├── fetch_car_repair_wiki.py # Wikipedia fetcher example
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Qdrant service
├── .env                 # Config (see below)
└── tests/               # Unit tests
```

## Setup
1. **Clone the repo and enter the directory.**
2. **Create and activate a Python 3.11+ virtual environment:**
   ```
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```
3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Start Qdrant with Docker Compose:**
   ```
   docker-compose up -d
   ```
5. **Create a `.env` file:**
   ```
   EMBEDDING_DIM=384
   COLLECTION_NAME=documents
   MODEL_NAME=all-MiniLM-L6-v2
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   DATA_DIR=data
   OPENAI_API_KEY=sk-...
   ```

## Ingest Data
Put your `.txt` files in the `data/` directory. Example: use `fetch_car_repair_wiki.py` to download Wikipedia articles about car repair.

Run ingestion:
```
python ingest.py
```

## Query
Run the query pipeline:
```
python query.py
```
You will be prompted for a question. Example:
```
Enter your query: What are minor car services?
```

## FastAPI Server
Run the API server:
```
uvicorn api:app --reload
```
- POST `/ingest`: Upload .txt files
- POST `/query`: Query the RAG pipeline
- HTTP Basic Auth required (set `API_USER` and `API_PASS` in `.env`)

## Troubleshooting
- **Vector dimension error:** Ensure `EMBEDDING_DIM` in `.env` matches your embedding model (384 for all-MiniLM-L6-v2)
- **Qdrant version warning:** Minor version mismatch is OK, but for best results use Qdrant 1.14.x (or update client)
- **OpenAI API error:** Use openai>=1.0.0 and ensure your API key is valid
- **KeyError 'content':** If you change the payload key in ingestion, update query.py to match

## Example Queries
- What are minor car services?
- How often should I change my oil?
- What does a vehicle inspection include?
- What is the role of an auto mechanic?

## License
MIT
