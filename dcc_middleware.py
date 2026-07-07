"""
Dynamic Context Compactor (DCC) — API Middleware for local LLM ecosystems.

Sits between the UI and the Main LLM:
  [User Prompt] -> (inject capsule) -> [Main LLM] -> (compact) -> [Vault]

Components:
  - Main LLM (Actor):      any OpenAI-compatible endpoint (Ollama /v1)
  - Memory Agent (Compactor): small model (Gemma 2B / Qwen 1.5B)
  - Vector Vault:          ChromaDB persistent store at ./dcc_memory/

Zero-leakage rule: raw chat logs are never persisted. Only the compressed
JSON capsule + the immediate new prompt ever reach the Main LLM.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional

import requests
from pydantic import BaseModel, Field, ValidationError

import chromadb


# ---------------------------------------------------------------------------
# 1. Capsule schema (enforced via pydantic)
# ---------------------------------------------------------------------------

class CapsuleMetadata(BaseModel):
    last_updated_frame: int = 0
    token_efficiency_saved: str = "0%"


class MemoryCapsule(BaseModel):
    topic: str
    global_context: str = ""
    key_decisions: list[str] = Field(default_factory=list)
    current_state: str = ""
    metadata: CapsuleMetadata = Field(default_factory=CapsuleMetadata)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def empty(cls, topic: str) -> "MemoryCapsule":
        return cls(
            topic=topic,
            global_context="New topic. No prior history.",
            current_state="Session just started.",
        )


# ---------------------------------------------------------------------------
# 2. Ollama / OpenAI-compatible client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Thin wrapper over an OpenAI-compatible endpoint (Ollama exposes /v1)
    plus Ollama's native /api/embeddings for local embedding models.
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, model: str, messages: list[dict],
             temperature: float = 0.7, json_mode: bool = False) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload, timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def embed(self, model: str, text: str) -> list[float]:
        r = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text}, timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["embedding"]


# ---------------------------------------------------------------------------
# 3. Vector Vault (ChromaDB persistent)
# ---------------------------------------------------------------------------

class VectorVault:
    def __init__(self, persist_dir: str = "./dcc_memory",
                 collection: str = "dcc_capsules"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def save(self, topic_id: str, capsule: MemoryCapsule,
             embedding: list[float]) -> None:
        """Overwrite the evolutionary capsule for this topic."""
        self._col.upsert(
            ids=[topic_id],
            embeddings=[embedding],
            documents=[capsule.to_json()],
            metadatas=[{"topic": capsule.topic,
                        "frame": capsule.metadata.last_updated_frame,
                        "updated_at": time.time()}],
        )

    def get(self, topic_id: str) -> Optional[MemoryCapsule]:
        """Direct fetch by topic id (exact continuity)."""
        res = self._col.get(ids=[topic_id])
        if not res["documents"]:
            return None
        return MemoryCapsule.model_validate_json(res["documents"][0])

    def query_relevant(self, embedding: list[float],
                       n: int = 1) -> Optional[MemoryCapsule]:
        """Semantic fetch: most relevant capsule to the incoming prompt."""
        if self._col.count() == 0:
            return None
        res = self._col.query(query_embeddings=[embedding], n_results=n)
        docs = res.get("documents") or [[]]
        if not docs[0]:
            return None
        return MemoryCapsule.model_validate_json(docs[0][0])


# ---------------------------------------------------------------------------
# 4. The Middleware
# ---------------------------------------------------------------------------

INJECTION_TEMPLATE = """[SYSTEM: SYSTEMIC PROJECT MEMORY CACHE]
You are a development partner with continuous episodic memory. Below is the ultra-dense status of the project/topic compiled from past historical frames:
---
{capsule_json}
---
Execute the following prompt focusing strictly on this context. Do not guess or hallucinate outside these boundaries."""

COMPACTOR_SYSTEM_PROMPT = """You are an advanced Memory Architect. Your sole job is to merge New Interactions into the Existing Project Memory Capsule.
You must maintain 100% integrity, absolute conciseness, and zero noise (remove greetings, chit-chat, and discarded error logs).
Update existing values if they changed. Add new decisions.

OUTPUT ONLY A VALID JSON OBJECT WITH THIS EXACT SCHEMA:
{
  "topic": "String - Project/Topic Title",
  "global_context": "String - High-level architectural overview and goal",
  "key_decisions": ["Array of Strings - Core architectural/logic decisions made so far"],
  "current_state": "String - Current active state of code, functions, or status",
  "metadata": { "last_updated_frame": "Integer", "token_efficiency_saved": "String" }
}"""


class ContextCompactorMiddleware:
    """
    Attach to any OpenAI-compatible endpoint. One call = full cycle:

        Phase A: embed prompt -> retrieve capsule -> inject -> Main LLM
        Phase B: Compactor merges (capsule + prompt + response) -> new capsule
                 -> re-embed -> upsert to vault. Raw logs discarded.
    """

    def __init__(
        self,
        main_model: str = "qwen3.5:7b-instruct",
        compactor_model: str = "qwen2.5:1.5b-instruct",
        embed_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        persist_dir: str = "./dcc_memory",
        client: Optional[OllamaClient] = None,
        on_event: Optional[Callable[[str], None]] = None,
    ):
        self.main_model = main_model
        self.compactor_model = compactor_model
        self.embed_model = embed_model
        self.client = client or OllamaClient(base_url)
        self.vault = VectorVault(persist_dir)
        self._log = on_event or (lambda msg: None)

    # ---- Public API -------------------------------------------------------

    def chat(self, topic_id: str, user_prompt: str) -> str:
        # ---------- Phase A: interception & injection ----------
        prompt_embedding = self.client.embed(self.embed_model, user_prompt)

        capsule = self.vault.get(topic_id)
        if capsule is None:
            capsule = self.vault.query_relevant(prompt_embedding) \
                      or MemoryCapsule.empty(topic_id)
        self._log(f"[Phase A] Injecting capsule (frame "
                  f"{capsule.metadata.last_updated_frame})")

        response = self.client.chat(
            self.main_model,
            messages=[
                {"role": "system",
                 "content": INJECTION_TEMPLATE.format(
                     capsule_json=capsule.to_json())},
                {"role": "user", "content": user_prompt},
            ],
        )

        # ---------- Phase B: compaction (blocking) ----------
        new_capsule = self._compact(capsule, user_prompt, response)
        new_embedding = self.client.embed(
            self.embed_model, new_capsule.to_json())
        self.vault.save(topic_id, new_capsule, new_embedding)
        self._log(f"[Phase B] Capsule evolved -> frame "
                  f"{new_capsule.metadata.last_updated_frame}")

        # Zero-leakage: nothing raw is retained beyond this scope.
        return response

    # ---- Internals --------------------------------------------------------

    def _compact(self, previous: MemoryCapsule, prompt: str,
                 response: str) -> MemoryCapsule:
        merge_input = (
            f"[EXISTING MEMORY CAPSULE]\n{previous.to_json()}\n\n"
            f"[NEW INTERACTION]\nUSER: {prompt}\nASSISTANT: {response}"
        )
        raw = self.client.chat(
            self.compactor_model,
            messages=[
                {"role": "system", "content": COMPACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": merge_input},
            ],
            temperature=0.1,
            json_mode=True,
        )
        capsule = self._parse_capsule(raw)
        if capsule is None:
            # Fail-safe: never lose continuity because a 1.5B model
            # produced malformed JSON. Keep old capsule, bump frame,
            # append raw state note.
            self._log("[Phase B] Compactor JSON invalid — fail-safe merge")
            capsule = previous.model_copy(deep=True)
            capsule.current_state = (
                f"{capsule.current_state} | (auto-note) latest turn about: "
                f"{prompt[:120]}"
            )
        capsule.metadata.last_updated_frame = (
            previous.metadata.last_updated_frame + 1
        )
        return capsule

    @staticmethod
    def _parse_capsule(raw: str) -> Optional[MemoryCapsule]:
        """Tolerant JSON extraction: strips fences / surrounding prose."""
        candidates = [raw.strip()]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
        if fenced:
            candidates.insert(0, fenced.group(1))
        braced = re.search(r"\{.*\}", raw, re.S)
        if braced:
            candidates.append(braced.group(0))
        for c in candidates:
            try:
                data = json.loads(c)
                # Coerce common small-model mistakes
                meta = data.get("metadata", {})
                if isinstance(meta.get("last_updated_frame"), str):
                    digits = re.sub(r"\D", "", meta["last_updated_frame"]) or "0"
                    meta["last_updated_frame"] = int(digits)
                return MemoryCapsule.model_validate(data)
            except (json.JSONDecodeError, ValidationError):
                continue
        return None
