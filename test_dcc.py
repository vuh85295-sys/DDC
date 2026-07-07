"""Offline verification: mocks OllamaClient, exercises the full cycle."""
import hashlib
import json
import shutil

from dcc_middleware import ContextCompactorMiddleware, MemoryCapsule


class MockClient:
    """Deterministic fake for OllamaClient."""
    def __init__(self):
        self.turn = 0

    def embed(self, model, text):
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:16]]  # 16-dim fake vector

    def chat(self, model, messages, temperature=0.7, json_mode=False):
        if json_mode:  # Compactor call
            self.turn += 1
            if self.turn == 2:
                # Simulate a sloppy small model: fenced JSON + prose
                inner = json.dumps({
                    "topic": "Todo API",
                    "global_context": "FastAPI + SQLite todo service",
                    "key_decisions": ["SQLAlchemy ORM", "pydantic v2",
                                      "POST /todos implemented"],
                    "current_state": "Todo model + POST endpoint done",
                    "metadata": {"last_updated_frame": "frame_2",
                                 "token_efficiency_saved": "78%"},
                })
                return f"Sure! Here is the JSON:\n```json\n{inner}\n```"
            if self.turn == 3:
                return "TOTALLY BROKEN {{{ not json"  # trigger fail-safe
            return json.dumps({
                "topic": "Todo API",
                "global_context": "FastAPI + SQLite todo service",
                "key_decisions": ["SQLAlchemy ORM", "pydantic v2"],
                "current_state": "Stack confirmed",
                "metadata": {"last_updated_frame": 1,
                             "token_efficiency_saved": "70%"},
            })
        # Main LLM call — verify injection happened
        sys_msg = messages[0]["content"]
        assert "SYSTEMIC PROJECT MEMORY CACHE" in sys_msg
        return f"(mock response, injected={'episodic memory' in sys_msg})"


def main():
    shutil.rmtree("./test_memory", ignore_errors=True)
    dcc = ContextCompactorMiddleware(
        client=MockClient(), persist_dir="./test_memory",
        on_event=print,
    )

    # Turn 1: cold start -> empty capsule injected, clean JSON compaction
    dcc.chat("t1", "Decide the stack")
    c = dcc.vault.get("t1")
    assert c and c.metadata.last_updated_frame == 1, c
    assert "SQLAlchemy ORM" in c.key_decisions

    # Turn 2: capsule retrieved, sloppy fenced JSON still parsed
    dcc.chat("t1", "Write POST /todos")
    c = dcc.vault.get("t1")
    assert c.metadata.last_updated_frame == 2
    assert "POST /todos implemented" in c.key_decisions
    assert c.metadata.token_efficiency_saved == "78%"

    # Turn 3: compactor returns garbage -> fail-safe keeps continuity
    dcc.chat("t1", "Add GET /todos")
    c = dcc.vault.get("t1")
    assert c.metadata.last_updated_frame == 3
    assert "auto-note" in c.current_state
    assert "POST /todos implemented" in c.key_decisions  # nothing lost

    # Persistence across process restart (new middleware instance)
    dcc2 = ContextCompactorMiddleware(client=MockClient(),
                                      persist_dir="./test_memory")
    c2 = dcc2.vault.get("t1")
    assert c2.metadata.last_updated_frame == 3

    # Semantic fallback path
    emb = MockClient().embed("x", "anything")
    assert dcc2.vault.query_relevant(emb) is not None

    print("\nALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
