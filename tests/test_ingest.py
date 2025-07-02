import tempfile
import pytest
from ingest import load_config, load_documents, split_text
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


def test_load_config(monkeypatch):
    monkeypatch.setenv('DATA_DIR', 'foo')
    cfg = load_config()
    assert cfg['DATA_DIR'] == 'foo'


def test_load_documents(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    f = d / "test.txt"
    f.write_text("hello world")
    docs = load_documents(str(d))
    assert docs == [("test.txt", "hello world")]


def test_load_documents_empty(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    docs = load_documents(str(d))
    assert docs == []


def test_split_text():
    text = "abcdefghij"
    chunks = split_text(text, chunk_size=4, overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]


def test_split_text_empty():
    assert split_text("", 4, 2) == []


def test_split_text_chunk_gt_text():
    assert split_text("abc", 10, 2) == []


def test_split_text_invalid():
    with pytest.raises(ValueError):
        split_text("abc", 4, 4)
    with pytest.raises(ValueError):
        split_text("abc", 0, 0)
    with pytest.raises(ValueError):
        split_text("abc", 4, -1)


def test_load_config_missing(monkeypatch):
    monkeypatch.delenv('DATA_DIR', raising=False)
    cfg = load_config()
    assert 'DATA_DIR' in cfg
