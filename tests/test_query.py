from query import load_config, EmbeddingModel, MilvusStore, OpenAIClient
from unittest.mock import patch, MagicMock
import pytest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


def test_load_config(monkeypatch):
    monkeypatch.setenv('COLLECTION_NAME', 'foo')
    cfg = load_config()
    assert cfg['COLLECTION_NAME'] == 'foo'


@patch('query.SentenceTransformer')
def test_embedding_model_embed(mock_st):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array(
        [[0.1, 0.2, 0.3]], dtype=np.float32)
    mock_st.return_value = mock_model
    model = EmbeddingModel('dummy')
    emb = model.embed('test')
    assert emb.shape == (1, 3)


def test_embedding_model_empty():
    model = EmbeddingModel('all-MiniLM-L6-v2')
    out = model.embed("")
    assert out.shape[0] == 1 or out.shape[0] == 0


@patch.object(MilvusStore, '_connect')
def test_milvus_store_search(mock_connect):
    mock_collection = MagicMock()
    mock_connect.return_value = mock_collection
    mock_collection.search.return_value = [
        [MagicMock(entity={'content': 'foo'})]]
    store = MilvusStore('localhost', '19530', 'col')
    res = store.search(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), top_k=1)
    assert res == ['foo']


def test_milvus_store_search_empty():
    store = MilvusStore.__new__(MilvusStore)
    store.collection = MagicMock()
    res = store.search(np.zeros((0, 3), dtype=np.float32), top_k=1)
    assert res == []


@patch('query.openai.ChatCompletion.create')
def test_openai_client_generate(mock_create):
    mock_create.return_value = MagicMock(
        choices=[MagicMock(message={'content': 'bar'})])
    client = OpenAIClient('sk-test', 'gpt', 10, 0.1)
    out = client.generate('ctx', 'q')
    assert out == 'bar'


def test_openai_client_no_key():
    with pytest.raises(ValueError):
        OpenAIClient('', 'gpt', 10, 0.1)
