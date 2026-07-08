"""Tests for POST /api/capsule/{topic_id} — capsule ingestion gate."""
import json
import pytest
from fastapi.testclient import TestClient
from dcc_middleware import MemoryCapsule, CapsuleMetadata

# ---------------------------------------------------------------------------
# Mocks — replace VectorVault and ollama_client to avoid ChromaDB/Ollama deps
# ---------------------------------------------------------------------------
class MockVault:
    _store: dict = {}

    def __init__(self, persist_dir=None):
        pass

    def get(self, topic_id):
        return self._store.get(topic_id, None)

    def save(self, topic_id, capsule, embedding):
        self._store[topic_id] = capsule


def mock_embed(model, text):
    return [0.0] * 16


# Patch at dcc_middleware level (since dcc_app does lazy imports from there)
import dcc_middleware
dcc_middleware.VectorVault = MockVault

import dcc_app
dcc_app.ollama_client.embed = mock_embed

from dcc_app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_vault():
    MockVault._store = {}
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ingest_valid_capsule_new_topic(client):
    """Nạp capsule hợp lệ vào topic mới → 201, GET trả đúng."""
    payload = {
        "topic": "kdm-test",
        "global_context": "Knowledge Domain Mapper — biến wish thành bản đồ hệ thống",
        "key_decisions": [
            "2 endpoint LLM: cloud map_maker + local expander",
            "JSON → Mermaid bằng code deterministic",
        ],
        "current_state": "V1.1 verified E2E — wish → map → duyệt → capsule",
        "metadata": {"last_updated_frame": 5, "token_efficiency_saved": "60%"},
    }
    resp = client.post("/api/capsule/kdm-test", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    capsule = data["capsule"]
    assert capsule["topic"] == "kdm-test"
    assert capsule["global_context"] == payload["global_context"]
    assert capsule["metadata"]["last_updated_frame"] == 0  # reset to Turn 0
    assert len(capsule["key_decisions"]) == 2

    # Verify via GET endpoint
    get_resp = client.get("/api/capsule/kdm-test")
    assert get_resp.status_code == 200
    stored = get_resp.json()["capsule"]
    assert stored["global_context"] == payload["global_context"]
    assert stored["metadata"]["last_updated_frame"] == 0


def test_ingest_duplicate_with_living_memory(client):
    """Nạp lần 2 sau khi đã có frame > 0 → 409."""
    # First — ingest a capsule (makes frame=0)
    payload = {
        "topic": "existing-topic",
        "global_context": "First capsule — Turn 0",
        "key_decisions": ["Initial decision"],
        "current_state": "Just started",
    }
    client.post("/api/capsule/existing-topic", json=payload)

    # Simulate frame > 0 by directly inserting a capsule with frame=1
    cap = MemoryCapsule(
        topic="existing-topic",
        global_context="Second capsule with frame=1",
        key_decisions=["Updated decision"],
        current_state="After chat",
        metadata=CapsuleMetadata(last_updated_frame=1),
    )
    MockVault._store["existing-topic"] = cap

    # Try to ingest again → should be 409
    resp = client.post("/api/capsule/existing-topic", json=payload)
    assert resp.status_code == 409, resp.text
    assert "living memory" in resp.json()["detail"].lower()


def test_ingest_topic_id_too_long(client):
    """topic_id 150 ký tự → 422."""
    long_id = "a" * 150
    payload = {
        "topic": "test",
        "global_context": "Some context",
        "current_state": "Starting",
    }
    resp = client.post(f"/api/capsule/{long_id}", json=payload)
    assert resp.status_code == 422, resp.text
    assert "3-63" in resp.json()["detail"]


def test_ingest_missing_global_context(client):
    """Body thiếu global_context → 422."""
    payload = {
        "topic": "no-context",
        "key_decisions": ["Some decision"],
        "current_state": "No global context",
    }
    resp = client.post("/api/capsule/no-context", json=payload)
    assert resp.status_code == 422, resp.text
    assert "global_context" in resp.json()["detail"].lower()
