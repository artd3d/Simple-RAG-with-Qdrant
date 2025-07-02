from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import List, Optional
from ingest import ingest_pipeline, load_config
from query import query_pipeline, load_config as load_query_config
import shutil
from pathlib import Path
import os
import secrets

app = FastAPI(title="RAG Milvus API")
security = HTTPBasic()

DATA_DIR = load_config().get('DATA_DIR', 'data')
API_USER = os.getenv('API_USER', 'admin')
API_PASS = os.getenv('API_PASS', 'admin')


def verify_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, API_USER)
    correct_password = secrets.compare_digest(credentials.password, API_PASS)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={
                            "WWW-Authenticate": "Basic"})


@app.post("/ingest")
def ingest_files(files: List[UploadFile] = File(...), creds: HTTPBasicCredentials = Depends(verify_basic_auth)):
    """Ingest uploaded .txt files into Milvus."""
    data_path = Path(DATA_DIR)
    data_path.mkdir(exist_ok=True)
    saved_files = []
    for file in files:
        if not file.filename.endswith('.txt'):
            continue
        dest = data_path / file.filename
        with dest.open('wb') as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(file.filename)
    if not saved_files:
        raise HTTPException(
            status_code=400, detail="No valid .txt files uploaded.")
    ingest_pipeline()
    return {"ingested": saved_files}


@app.post("/query")
def query_api(query: str = Form(...), creds: HTTPBasicCredentials = Depends(verify_basic_auth)):
    """Query the RAG pipeline and get an LLM answer."""
    answer = query_pipeline(user_query=query)
    if answer is None:
        return JSONResponse(status_code=404, content={"answer": None, "detail": "No relevant context found."})
    return {"answer": answer}


@app.get("/")
def root():
    return {"msg": "RAG Milvus API. Use /ingest (POST) and /query (POST)."}
