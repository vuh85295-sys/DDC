"""DCC Web App — FastAPI server with topic management + multi-provider + DCC memory."""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from dcc_middleware import (
    ContextCompactorMiddleware,
    MemoryCapsule,
    OllamaClient,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 8888
DCC_PERSIST_DIR = "./dcc_memory"

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
class ProviderInfo(BaseModel):
    id: str
    name: str
    type: str
    models: list[str]

PROVIDERS = [
    ProviderInfo(id="ollama", name="Ollama (Local)", type="local",
                 models=["qwen2.5:7b-instruct", "qwen2.5:1.5b-instruct", "qwen3.5:9b", "gemma4:e4b-mlx"]),
    ProviderInfo(id="openai", name="OpenAI", type="remote",
                 models=["gpt-4o", "gpt-4o-mini"]),
    ProviderInfo(id="anthropic", name="Anthropic (Claude)", type="remote",
                 models=["claude-sonnet-4", "claude-haiku-3.5"]),
    ProviderInfo(id="google", name="Google (Gemini)", type="remote",
                 models=["gemini-2.0-flash", "gemini-2.5-pro"]),
    ProviderInfo(id="xai", name="xAI (Grok)", type="remote",
                 models=["grok-3", "grok-3-mini"]),
]

# ---------------------------------------------------------------------------
# Remote clients
# ---------------------------------------------------------------------------
class RemoteClient:
    @staticmethod
    def chat(provider_id: str, model: str, messages: list[dict], api_key: str, base_url: Optional[str] = None) -> str:
        fns = {"openai": RemoteClient._openai, "anthropic": RemoteClient._anthropic,
               "google": RemoteClient._gemini, "xai": RemoteClient._xai}
        return fns[provider_id](model, messages, api_key, base_url)

    @staticmethod
    def _openai(model, msgs, key, url):
        url = (url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        r = requests.post(url, json={"model": model, "messages": msgs},
                          headers={"Authorization": f"Bearer {key}"}, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _anthropic(model, msgs, key, _url):
        system = ""
        converted = []
        for m in msgs:
            if m["role"] == "system":
                system = m["content"]
            else:
                converted.append({"role": m["role"], "content": m["content"]})
        payload = {"model": model, "max_tokens": 4096, "messages": converted}
        if system:
            payload["system"] = system
        r = requests.post("https://api.anthropic.com/v1/messages", json=payload,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"}, timeout=120)
        r.raise_for_status()
        return r.json()["content"][0]["text"]

    @staticmethod
    def _gemini(model, msgs, key, _url):
        contents = []
        sys_text = ""
        for m in msgs:
            if m["role"] == "system":
                sys_text = m["content"]
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        payload = {"contents": contents}
        if sys_text:
            payload["systemInstruction"] = {"parts": [{"text": sys_text}]}
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    @staticmethod
    def _xai(model, msgs, key, _url):
        r = requests.post("https://api.x.ai/v1/chat/completions",
                          json={"model": model, "messages": msgs},
                          headers={"Authorization": f"Bearer {key}"}, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------------
# Topic store (in-memory chat history per topic)
# ---------------------------------------------------------------------------
TOPIC_STORE: dict[str, list[dict]] = {}  # topic_id -> [{"role","content"}, ...]

def get_history(topic_id: str) -> list[dict]:
    return TOPIC_STORE.setdefault(topic_id, [])

# ---------------------------------------------------------------------------
# DCC Engine
# ---------------------------------------------------------------------------
ollama_client = OllamaClient()

# KDM-style language contract templates
LANG_CONTRACT_HEAD = """## KHẾ ƯỚC NGÔN NGỮ — BẮT BUỘC
- Trả lời 100% bằng tiếng Việt có dấu, trừ code và thuật ngữ chuyên ngành.
- Đúng: "Edge Computing — Xử lý dữ liệu gần nguồn..."
- Sai: "Edge Computing is about processing data near the source..."
"""

LANG_CONTRACT_TAIL = """
## NHẮC LẠI KHẾ ƯỚC NGÔN NGỮ
- Trả lời bằng tiếng Việt có dấu (code/terms giữ nguyên gốc).
- Output vi phạm sẽ bị phát hiện và từ chối.
"""

# Actor constitution — enforcement rules placed AFTER capsule (model primacy)
ANTI_MAP_RULE = """⛔ 1. Yêu cầu trúng mục "Ngoài phạm vi (anti-map)" trong capsule → KHÔNG thiết kế, KHÔNG viết code.
   Trả lời: nhắc mục anti-map + lý do nguyên văn, rồi hỏi:
   "Muốn đưa vào phạm vi? Cập nhật bản đồ KDM trước."""

RED_DECISION_RULE = """⛔ 2. Đề xuất thay đổi quyết định có mức Reversibility 🔴 → KHÔNG đồng ý, KHÔNG tự quyết.
   Được phân tích trade-off, nhưng phải kết: quyết định chỉ mở lại khi user XÁC NHẬN
   đã chạm Điểm chuyển đổi (switch_trigger) — trích nguyên văn từ capsule."""

NO_FAKE_MEMORY_RULE = """⛔ 3. CẤM bịa trạng thái dự án khác current_state trong capsule.
   CẤM tự sinh block [SYSTEM:...] hay tuyên bố "đã ghi vào bộ nhớ" — chỉ Compactor được ghi."""

NO_CHEERLEAD_RULE = """⛔ 4. KHÔNG mở đầu bằng khen đề xuất ("hợp lý", "rất hay") trước khi đối chiếu hiến pháp."""


def build_actor_system_prompt(capsule: MemoryCapsule) -> str:
    """Build a constitution-enforcing system prompt from the capsule.
    
    Empty capsule → returns a plain chat prompt without constitution.
    Active capsule → wraps capsule in HIẾN PHÁP THI HÀNH with enforcement rules.
    """
    # Detect empty capsule (cold start — no real project memory)
    is_empty = (
        not capsule.global_context.strip()
        or capsule.global_context == "New topic. No prior history."
    ) and len(capsule.key_decisions) == 0

    if is_empty:
        return (
            f"{LANG_CONTRACT_HEAD}\n"
            "Bạn là trợ lý AI thông thường. Hãy trò chuyện tự nhiên.\n"
            f"{LANG_CONTRACT_TAIL}"
        )

    # Build capsule section
    capsule_block = f"<capsule>\n{capsule.to_json()}\n</capsule>"

    # Build enforcement rules (only include relevant ones)
    rules = [ANTI_MAP_RULE, RED_DECISION_RULE, NO_FAKE_MEMORY_RULE, NO_CHEERLEAD_RULE]

    rules_block = "\n\n".join(rules)

    return f"""## HIẾN PHÁP DỰ ÁN — BẤT BIẾN

Bạn là kiến trúc sư của dự án, vận hành DƯỚI hiến pháp bất biến dưới đây.
Nhiệm vụ của bạn là BẢO VỆ các quyết định đã chốt, không phải chiều theo mọi đề xuất.

---

{capsule_block}

---

## LUẬT THI HÀNH

{rules_block}

---

{LANG_CONTRACT_HEAD}
{LANG_CONTRACT_TAIL}"""

def _compact(capsule: MemoryCapsule, prompt: str, response: str, topic_id: str) -> MemoryCapsule:
    """Compactor dùng qwen2.5:7b-instruct (chính xác hơn 1.5b)."""
    merge_input = f"[EXISTING CAPSULE]\n{capsule.to_json()}\n\n[NEW INTERACTION]\nUSER: {prompt}\nASSISTANT: {response}"
    try:
        raw = ollama_client.chat("qwen2.5:7b-instruct", [
            {"role": "system", "content": (
                "Bạn là Memory Architect. Nhiệm vụ: merge tương tác mới vào capsule cũ.\n"
                "CHỈ xuất ra JSON hợp lệ, không thêm text nào khác.\n"
                "Schema:\n"
                '{"topic": "...", "global_context": "...", "key_decisions": ["..."],\n'
                ' "current_state": "...", "metadata": {"last_updated_frame": 0, "token_efficiency_saved": "0%"}}'
            )},
            {"role": "user", "content": merge_input},
        ], temperature=0.05, json_mode=True)
        found = re.search(r"\{.*\}", raw, re.S)
        if found:
            data = json.loads(found.group(0))
            new_cap = MemoryCapsule.model_validate(data)
            new_cap.metadata.last_updated_frame = capsule.metadata.last_updated_frame + 1
            # Save to vault
            from dcc_middleware import VectorVault
            vault = VectorVault(DCC_PERSIST_DIR)
            try:
                emb = ollama_client.embed("nomic-embed-text", new_cap.to_json())
            except Exception:
                emb = [0.0] * 16
            vault.save(topic_id, new_cap, emb)
            return new_cap
    except Exception as e:
        print(f"[COMPACTOR] Error: {e}")

    # Fail-safe
    capsule.metadata.last_updated_frame += 1
    capsule.current_state = f"{capsule.current_state} | (turn: {prompt[:80]})"
    try:
        from dcc_middleware import VectorVault
        vault = VectorVault(DCC_PERSIST_DIR)
        emb = ollama_client.embed("nomic-embed-text", capsule.to_json())
        vault.save(topic_id, capsule, emb)
    except Exception:
        pass
    return capsule

def dcc_chat(topic_id: str, prompt: str, provider_id: str, model: str,
             api_key: str = "", base_url: str = "") -> dict:
    from dcc_middleware import VectorVault
    vault = VectorVault(DCC_PERSIST_DIR)

    # Embed prompt
    try:
        prompt_emb = ollama_client.embed("nomic-embed-text", prompt)
    except Exception:
        prompt_emb = [0.0] * 16

    # Retrieve capsule
    capsule = vault.get(topic_id)
    if capsule is None:
        capsule = MemoryCapsule.empty(topic_id)

    # Build messages: capsule injection + hiến pháp thi hành
    inject = build_actor_system_prompt(capsule)
    messages = [
        {"role": "system", "content": inject},
        {"role": "user", "content": prompt},
    ]

    # Call provider
    try:
        if provider_id == "ollama":
            response = ollama_client.chat(model, messages)
        else:
            response = RemoteClient.chat(provider_id, model, messages, api_key, base_url)
    except Exception as e:
        return {"error": str(e), "capsule": json.loads(capsule.to_json()) if capsule else None}

    # Compact
    new_cap = _compact(capsule, prompt, response, topic_id)
    final_cap = json.loads(new_cap.to_json())

    # Store chat history
    get_history(topic_id).append({"role": "user", "content": prompt})
    get_history(topic_id).append({"role": "assistant", "content": response})

    return {"response": response, "capsule": final_cap}

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="DCC Web")

class ChatRequest(BaseModel):
    topic_id: str = "default"
    prompt: str
    provider_id: str = "ollama"
    model: str = "qwen2.5:7b-instruct"
    api_key: str = ""
    base_url: str = ""

class NewTopicRequest(BaseModel):
    topic_id: str
    description: str = ""

@app.get("/api/providers")
def list_providers():
    return PROVIDERS

@app.get("/api/topics")
def list_topics():
    """List all topics that have capsule data in vault."""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=DCC_PERSIST_DIR)
        col = client.get_or_create_collection("dcc_capsules")
        all_data = col.get()
        topics = []
        if all_data["ids"]:
            for i, tid in enumerate(all_data["ids"]):
                meta = (all_data["metadatas"] or [{}])[i] or {}
                doc = all_data["documents"][i] if all_data["documents"] else "{}"
                try:
                    doc_json = json.loads(doc) if doc else {}
                except Exception:
                    doc_json = {}
                topics.append({
                    "id": tid,
                    "topic": doc_json.get("topic", tid),
                    "frame": meta.get("frame", 0),
                    "updated": meta.get("updated_at", 0),
                    "msg_count": len(get_history(tid)),
                })
        topics.sort(key=lambda t: t["updated"], reverse=True)
        return {"topics": topics}
    except Exception:
        return {"topics": []}

@app.post("/api/topics")
def create_topic(req: NewTopicRequest):
    tid = req.topic_id.strip().replace(" ", "_").lower()
    if not tid:
        raise HTTPException(400, "Invalid topic_id")
    # Create empty capsule
    from dcc_middleware import VectorVault
    vault = VectorVault(DCC_PERSIST_DIR)
    cap = MemoryCapsule.empty(tid)
    if req.description:
        cap.global_context = req.description
    try:
        emb = ollama_client.embed("nomic-embed-text", cap.to_json())
    except Exception:
        emb = [0.0] * 16
    vault.save(tid, cap, emb)
    return {"topic_id": tid, "status": "created"}

@app.delete("/api/topics/{topic_id}")
def delete_topic(topic_id: str):
    """Clear chat history + capsule for a topic."""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=DCC_PERSIST_DIR)
        col = client.get_or_create_collection("dcc_capsules")
        col.delete(ids=[topic_id])
    except Exception:
        pass
    TOPIC_STORE.pop(topic_id, None)
    return {"status": "deleted"}

@app.get("/api/capsule/{topic_id}")
def get_capsule(topic_id: str):
    from dcc_middleware import VectorVault
    vault = VectorVault(DCC_PERSIST_DIR)
    c = vault.get(topic_id)
    return {"capsule": json.loads(c.to_json()) if c else None}


@app.post("/api/capsule/{topic_id}", status_code=201)
def ingest_capsule(topic_id: str, capsule: MemoryCapsule):
    """Ingest a capsule from external systems (KDM ecosystem).
    
    Validates topic_id format, checks for existing living memory,
    then saves capsule as Turn 0.
    """
    # Validate topic_id format: 3-63 chars, lowercase ascii + digits + hyphens
    if not re.match(r"^[a-z0-9-]{3,63}$", topic_id):
        raise HTTPException(
            422,
            "topic_id must be 3-63 characters, lowercase ASCII letters, "
            "digits, and hyphens only.",
        )

    from dcc_middleware import VectorVault
    vault = VectorVault(DCC_PERSIST_DIR)

    # Check existing living memory (frame > 0)
    existing = vault.get(topic_id)
    if existing and existing.metadata.last_updated_frame > 0:
        raise HTTPException(
            409,
            f"Topic already has living memory (frame "
            f"{existing.metadata.last_updated_frame}). Cannot overwrite. "
            f"Use a different topic_id or delete the topic first.",
        )

    # Require global_context (not just empty default)
    if not capsule.global_context.strip():
        raise HTTPException(
            422,
            "global_context is required and cannot be empty.",
        )

    # Override topic field and reset frame to Turn 0
    capsule.topic = topic_id
    capsule.metadata.last_updated_frame = 0

    # Embed and save
    try:
        emb = ollama_client.embed("nomic-embed-text", capsule.to_json())
    except Exception:
        emb = [0.0] * 16
    vault.save(topic_id, capsule, emb)

    return {"capsule": json.loads(capsule.to_json())}


@app.get("/api/chat/{topic_id}")
def get_topic_history(topic_id: str):
    return {"history": get_history(topic_id)}

@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is required")
    result = dcc_chat(
        topic_id=req.topic_id, prompt=req.prompt,
        provider_id=req.provider_id, model=req.model,
        api_key=req.api_key, base_url=req.base_url,
    )
    if "error" in result:
        raise HTTPException(502, result["error"])
    return result

# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML_PAGE)

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DCC — Chat với AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz@14..32&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --accent-hover: #79c0ff;
    --green: #3fb950;
    --orange: #d29922;
    --red: #f85149;
    --radius: 8px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Inter', -apple-system, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }
  .header { padding: 10px 16px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; background: var(--surface); flex-shrink: 0; }
  .header h1 { font-size: 16px; font-weight: 600; }
  .header .logo { color: var(--accent); font-size: 20px; }
  .tag { font-size: 10px; color: var(--text-dim); background: #21262d; padding: 2px 8px; border-radius: 4px; }
  .header-controls { display: flex; gap: 8px; align-items: center; margin-left: auto; flex-wrap: wrap; }
  .header-controls select, .header-controls input { background: #21262d; border: 1px solid var(--border); color: var(--text); padding: 5px 8px; border-radius: var(--radius); font-size: 12px; }
  .header-controls select { min-width: 120px; }
  .header-controls input[type=password] { min-width: 140px; }
  .layout { display: flex; flex: 1; overflow: hidden; }
  .sidebar { width: 240px; border-right: 1px solid var(--border); display: flex; flex-direction: column; background: var(--surface); flex-shrink: 0; }
  .sidebar-header { padding: 10px 12px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
  .sidebar-header h3 { font-size: 11px; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.5px; }
  .topic-list { flex: 1; overflow-y: auto; padding: 4px 0; }
  .topic-item { padding: 8px 12px; cursor: pointer; display: flex; align-items: center; gap: 8px; border-left: 3px solid transparent; font-size: 13px; }
  .topic-item:hover { background: #1c2128; }
  .topic-item.active { background: #1c2128; border-left-color: var(--accent); }
  .topic-item .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--border); flex-shrink: 0; }
  .topic-item .dot.active-dot { background: var(--green); }
  .topic-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .topic-item .count { font-size: 11px; color: var(--text-dim); }
  .topic-item .del-btn { opacity: 0; font-size: 14px; color: var(--text-dim); background: none; border: none; cursor: pointer; padding: 0 2px; }
  .topic-item:hover .del-btn { opacity: 1; }
  .topic-item .del-btn:hover { color: var(--red); }
  .sidebar-footer { padding: 8px 12px; border-top: 1px solid var(--border); display: flex; gap: 6px; }
  .sidebar-footer input { flex: 1; background: #0d1117; border: 1px solid var(--border); color: var(--text); padding: 6px 8px; border-radius: var(--radius); font-size: 12px; }
  .sidebar-footer button { background: var(--accent); color: #000; border: none; border-radius: var(--radius); padding: 6px 12px; font-size: 12px; font-weight: 600; cursor: pointer; white-space: nowrap; }
  .main-area { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .topic-label { padding: 6px 16px; border-bottom: 1px solid var(--border); font-size: 12px; color: var(--text-dim); background: var(--surface); flex-shrink: 0; display: flex; align-items: center; gap: 6px; }
  .topic-label .badge { background: #21262d; padding: 1px 6px; border-radius: 3px; font-size: 10px; }
  .messages { flex: 1; overflow-y: auto; padding: 12px 16px; display: flex; flex-direction: column; gap: 10px; }
  .msg { max-width: 80%; padding: 10px 14px; border-radius: var(--radius); line-height: 1.5; font-size: 14px; white-space: pre-wrap; word-break: break-word; }
  .msg.user { background: #1f6feb; align-self: flex-end; border-bottom-right-radius: 2px; }
  .msg.assistant { background: #21262d; align-self: flex-start; border-bottom-left-radius: 2px; }
  .msg.system { background: transparent; align-self: center; font-size: 11px; color: var(--text-dim); border: 1px dashed var(--border); padding: 6px 16px; text-align: center; }
  .msg code { background: #0d1117; padding: 1px 4px; border-radius: 3px; font-size: 13px; }
  .msg pre { background: #0d1117; padding: 8px; border-radius: 4px; margin: 6px 0; overflow-x: auto; font-size: 13px; }
  .input-area { padding: 10px 16px; border-top: 1px solid var(--border); display: flex; gap: 8px; background: var(--surface); flex-shrink: 0; }
  .input-area textarea { flex: 1; background: #0d1117; border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); padding: 8px 12px; font-size: 14px; font-family: inherit; resize: none; min-height: 40px; max-height: 100px; }
  .input-area textarea:focus { outline: none; border-color: var(--accent); }
  .input-area button { background: var(--accent); color: #000; border: none; border-radius: var(--radius); padding: 8px 18px; font-size: 14px; font-weight: 600; cursor: pointer; }
  .input-area button:disabled { opacity: 0.4; cursor: not-allowed; }
  .capsule-panel { width: 260px; border-left: 1px solid var(--border); padding: 10px; overflow-y: auto; background: var(--surface); display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
  .capsule-panel h3 { font-size: 11px; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.5px; }
  .capsule-content { flex: 1; overflow-y: auto; background: #0d1117; border: 1px solid var(--border); border-radius: var(--radius); padding: 8px; font-size: 11px; line-height: 1.5; font-family: 'SF Mono', monospace; color: var(--text-dim); white-space: pre-wrap; }
  .capsule-empty { color: var(--text-dim); font-style: italic; font-size: 11px; padding: 16px; text-align: center; }
  .loading { display: flex; align-items: center; gap: 8px; color: var(--text-dim); font-size: 13px; padding: 8px; }
  .spinner { width: 14px; height: 14px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  @media (max-width: 900px) { .capsule-panel { display: none; } .sidebar { width: 180px; } }
  @media (max-width: 640px) { .sidebar { display: none; } }
  .hl-key { color: var(--accent); }
  .hl-str { color: var(--green); }
  .hl-num { color: var(--orange); }
</style>
</head>
<body>

<div class="header">
  <span class="logo">◈</span>
  <h1>DCC</h1>
  <span class="tag">Dynamic Context Compactor</span>
  <div class="header-controls">
    <select id="provider" onchange="onProviderChange()">
      <option value="ollama">Ollama (Local)</option>
      <option value="openai">OpenAI</option>
      <option value="anthropic">Claude</option>
      <option value="google">Gemini</option>
      <option value="xai">Grok</option>
    </select>
    <select id="model"><option>qwen2.5:7b-instruct</option></select>
    <input id="api_key" type="password" placeholder="API Key">
    <input id="base_url" type="text" placeholder="Base URL" style="width:120px;display:none">
  </div>
</div>

<div class="layout">
  <!-- LEFT: Topics -->
  <div class="sidebar">
    <div class="sidebar-header"><h3>📋 Chủ đề</h3></div>
    <div id="topicList" class="topic-list"></div>
    <div class="sidebar-footer">
      <input id="newTopicInput" placeholder="Tên chủ đề mới..." onkeydown="if(event.key==='Enter')createTopic()">
      <button onclick="createTopic()">+</button>
    </div>
  </div>

  <!-- CENTER: Chat -->
  <div class="main-area">
    <div class="topic-label">
      🧵 <span id="currentTopicLabel">default</span>
      <span id="msgCount" class="badge">0 tin nhắn</span>
    </div>
    <div id="messages" class="messages">
      <div class="msg system">👋 Chào anh Ruka! Chọn chủ đề bên trái và bắt đầu chat. DCC tự động ghi nhớ toàn bộ context.</div>
    </div>
    <div class="input-area">
      <textarea id="input" rows="1" placeholder="Nhập tin nhắn..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
      <button id="sendBtn" onclick="send()">Gửi</button>
    </div>
  </div>

  <!-- RIGHT: Capsule -->
  <div class="capsule-panel">
    <h3>🧠 Memory Capsule</h3>
    <div id="capsuleView" class="capsule-content"><div class="capsule-empty">Chọn chủ đề để xem capsule</div></div>
  </div>
</div>

<script>
const PROVIDER_MODELS = {
  ollama: ["qwen2.5:7b-instruct","qwen2.5:1.5b-instruct","qwen3.5:9b","gemma4:e4b-mlx"],
  openai: ["gpt-4o","gpt-4o-mini","gpt-4-turbo"],
  anthropic: ["claude-sonnet-4","claude-haiku-3.5"],
  google: ["gemini-2.0-flash","gemini-2.5-pro"],
  xai: ["grok-3","grok-3-mini"],
};
const PROVIDER_NEEDS_KEY = {ollama:false, openai:true, anthropic:true, google:true, xai:true};

let currentTopic = 'default';
let loading = false;

function onProviderChange() {
  const p = document.getElementById('provider').value;
  const sel = document.getElementById('model');
  sel.innerHTML = (PROVIDER_MODELS[p]||["unknown"]).map(m => '<option value="'+m+'">'+m+'</option>').join('');
  document.getElementById('api_key').placeholder = PROVIDER_NEEDS_KEY[p] ? 'API Key (required)' : 'API Key (optional)';
  document.getElementById('base_url').style.display = p === 'openai' ? 'inline-block' : 'none';
}
onProviderChange();

async function loadTopics() {
  const res = await fetch('/api/topics');
  const data = await res.json();
  const list = document.getElementById('topicList');
  const topics = data.topics || [];
  if (topics.length === 0) {
    list.innerHTML = '<div style="padding:16px;font-size:12px;color:var(--text-dim);text-align:center">Chưa có chủ đề<br><span style="font-size:11px">Tạo chủ đề mới bên dưới</span></div>';
    return;
  }
  list.innerHTML = topics.map(t =>
    `<div class="topic-item ${t.id === currentTopic ? 'active' : ''}" onclick="switchTopic('${t.id}')">
      <span class="dot ${t.msg_count > 0 ? 'active-dot' : ''}"></span>
      <span class="name">${t.topic || t.id}</span>
      <span class="count">${t.msg_count}</span>
      <button class="del-btn" onclick="event.stopPropagation();deleteTopic('${t.id}')">×</button>
    </div>`
  ).join('');
}

async function switchTopic(id) {
  currentTopic = id;
  document.getElementById('currentTopicLabel').textContent = id;
  // Load history
  const histRes = await fetch('/api/chat/' + encodeURIComponent(id));
  const hist = await histRes.json();
  const msgs = document.getElementById('messages');
  msgs.innerHTML = '';
  if (hist.history && hist.history.length > 0) {
    for (const m of hist.history) {
      const el = document.createElement('div');
      el.className = 'msg ' + m.role;
      el.textContent = m.content;
      msgs.appendChild(el);
    }
  } else {
    msgs.innerHTML = '<div class="msg system">💬 Chủ đề mới. Hãy gửi tin nhắn đầu tiên!</div>';
  }
  document.getElementById('msgCount').textContent = (hist.history?.length||0)/2 + ' tin nhắn';
  // Load capsule
  loadCapsule(id);
  loadTopics();
  msgs.scrollTop = msgs.scrollHeight;
}

async function loadCapsule(id) {
  const res = await fetch('/api/capsule/' + encodeURIComponent(id));
  const data = await res.json();
  const view = document.getElementById('capsuleView');
  if (!data.capsule) {
    view.innerHTML = '<div class="capsule-empty">Chưa có dữ liệu</div>';
    return;
  }
  view.innerHTML = syntaxHighlight(JSON.stringify(data.capsule, null, 2));
}

async function createTopic() {
  const input = document.getElementById('newTopicInput');
  const name = input.value.trim();
  if (!name) return;
  await fetch('/api/topics', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({topic_id: name})});
  input.value = '';
  await switchTopic(name.replace(/\\s+/g,'_').toLowerCase());
}

async function deleteTopic(id) {
  if (!confirm('Xoá chủ đề "'+id+'"?')) return;
  await fetch('/api/topics/' + encodeURIComponent(id), {method:'DELETE'});
  if (currentTopic === id) switchTopic('default');
  else loadTopics();
}

async function send() {
  if (loading) return;
  const input = document.getElementById('input');
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  input.style.height = 'auto';

  addMsg('user', prompt);
  showLoading();
  loading = true;
  document.getElementById('sendBtn').disabled = true;

  try {
    const provider = document.getElementById('provider').value;
    const model = document.getElementById('model').value;
    const api_key = document.getElementById('api_key').value;
    const base_url = document.getElementById('base_url').value;

    const res = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({topic_id:currentTopic, prompt, provider_id:provider, model, api_key, base_url}),
    });
    const data = await res.json();
    hideLoading();
    if (res.ok && data.response) {
      addMsg('assistant', data.response);
      if (data.capsule) {
        document.getElementById('capsuleView').innerHTML = syntaxHighlight(JSON.stringify(data.capsule, null, 2));
      }
    } else {
      addMsg('system', '❌ Lỗi: ' + (data.detail || 'Unknown'));
    }
  } catch(e) {
    hideLoading();
    addMsg('system', '❌ Kết nối thất bại: ' + e.message);
  }
  loading = false;
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('msgCount').textContent = document.getElementById('messages').querySelectorAll('.msg.user').length + ' tin nhắn';
  loadTopics();
}

function addMsg(role, content) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.textContent = content;
  document.getElementById('messages').appendChild(el);
  el.scrollIntoView({behavior:'smooth'});
}

function showLoading() {
  const el = document.createElement('div');
  el.id = 'loading'; el.className = 'loading';
  el.innerHTML = '<div class="spinner"></div> Đang xử lý...';
  document.getElementById('messages').appendChild(el);
  el.scrollIntoView({behavior:'smooth'});
}

function hideLoading() {
  const el = document.getElementById('loading');
  if (el) el.remove();
}

function syntaxHighlight(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"([^"]+)":/g, '<span class="hl-key">"$1"</span>:')
    .replace(/"([^"]*)"(,?$)/gm, (m,p1,p2) => '<span class="hl-str">"'+p1+'"</span>'+(p2||''))
    .replace(/\b(\d+(\.\d+)?)\b/g, '<span class="hl-num">$1</span>');
}

document.getElementById('input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 100) + 'px';
});

// Auto-load default topic
switchTopic('default');
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    print(f"\n  ◈ DCC Web — http://localhost:{PORT}")
    print(f"  Providers: Ollama, OpenAI, Claude, Gemini, Grok")
    print(f"  Topics: sidebar trái, tạo/xoá chủ đề riêng, chat history per topic\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
