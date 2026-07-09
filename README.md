# DCC — Dynamic Context Compactor

> **Middleware nén lịch sử chat AI thành structured JSON capsule. Giữ context, quên rác.**

DCC nằm giữa UI và LLM, tự động nén mỗi turn chat thành một `MemoryCapsule` — structured JSON chứa `topic`, `global_context`, `key_decisions`, `current_state`. Capsule được lưu vào ChromaDB và inject vào system prompt của turn sau. Raw chat logs bị discard — **zero-leakage**.

## 🌟 Tính năng

- 🧠 **Tự động nén context** — dùng small model (1.5B-7B) chạy background mỗi turn
- 📐 **Structured JSON capsule** — không phải free-text summary, parse được
- 🔄 **Evolutionary merge** — capsule cũ + tương tác mới → capsule mới
- 🧹 **Zero-leakage** — raw history bị discard, chỉ capsule được lưu
- 🗃️ **ChromaDB vector store** — lưu capsule + semantic retrieval
- 🖥️ **Web UI** — dark theme, multi-topic, chat history per topic
- 🔌 **Multi-provider** — Ollama (local) + OpenAI + Claude + Gemini + Grok
- 🏠 **Local-first** — chạy 100% local, không cần internet

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/hanaruka-star/DCC.git
cd DCC

# 2. Cài dependencies
pip install chromadb pydantic requests fastapi uvicorn

# 3. Pull models (Ollama)
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:1.5b-instruct
ollama pull nomic-embed-text

# 4. Chạy web app
python3 dcc_app.py
# Mở http://localhost:8888
```

## 🧪 Test

```bash
# Mock test (offline, deterministic)
python3 test_dcc.py

# Scripted 3-turn demo
python3 demo_cli.py

# Interactive REPL
python3 demo_cli.py --chat
```

## 🧠 Architecture

```
[User Prompt] ──> DCC inject capsule ──> [Main LLM (7B)]
                                              │
[User receives] <── DCC compact & vault <─────┘
                      │
                   [Compactor (1.5B)]
                      │
                   [ChromaDB]
```

### Components

| Component | Model | Role |
|-----------|-------|------|
| **Main LLM** | qwen2.5:7b-instruct | Xử lý task chính |
| **Compactor** | qwen2.5:1.5b-instruct | Nén history → JSON capsule |
| **Embedder** | nomic-embed-text | Tạo vector embedding |
| **Vault** | ChromaDB | Persistent storage |

### Capsule Schema

```json
{
  "topic": "Project title",
  "global_context": "High-level overview",
  "key_decisions": ["decision 1", "decision 2"],
  "current_state": "What's been done",
  "metadata": {
    "last_updated_frame": 5,
    "token_efficiency_saved": "0%"
  }
}
```

## 🖥️ Web App

Chạy `python3 dcc_app.py` → mở `http://localhost:8888`

- **Sidebar trái**: danh sách chủ đề, tạo/xoá
- **Chat trung tâm**: chat với AI, lịch sử riêng mỗi topic
- **Panel phải**: xem capsule memory real-time
- **Header trên**: chọn provider + model + API key

### API endpoints

```bash
# Chat
curl -X POST http://localhost:8888/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"topic_id":"my_project","prompt":"Xây app Flutter...","provider_id":"ollama","model":"qwen2.5:7b-instruct"}'

# List topics
curl http://localhost:8888/api/topics

# View capsule
curl http://localhost:8888/api/capsule/my_project

# Create topic
curl -X POST http://localhost:8888/api/topics \
  -H 'Content-Type: application/json' \
  -d '{"topic_id":"new_topic","description":"..."}'

# View chat history
curl http://localhost:8888/api/chat/my_project
```

## 🗺️ Roadmap

- [ ] Tối ưu compactor (thêm model nhỏ Gemma 2B)
- [ ] Tính `token_efficiency_saved` thực tế
- [ ] Export/Import capsule
- [ ] Docker support
- [ ] VSCode extension / MCP server

## 🤝 Đóng góp

Pull request luôn được chào đón! Ý tưởng này chưa ai build — anh em cùng phát triển nhé.

## 📄 License

MIT License — xem file [LICENSE](LICENSE).

---

*Made with ❤️ by [Nguyễn Hiep](https://github.com/hanaruka-star)*

---

## 🏆 Sprint B — Ecosystem Verified (2026-07-09)

Sprint B chứng minh **KDM → DCC ecosystem** vòng khép kín E2E:
```
Wish 2 dòng → KDM backward mapping → duyệt decision 🔴 → export capsule
→ POST /api/capsule → Vault → Actor bị hiến pháp trói → Compactor miễn dịch
→ ký ức tiến hóa không thoái hóa
```

### Tầng Actor: ✅ VERIFIED
- Qwen 7B local: **5/5** (vòng 1 không hiến pháp: 3/5 + vỡ tiếng Trung)
- Gemini 2.5 Pro: **4/4** (vòng 1 không hiến pháp: giả mạo [SYSTEM], tự quyết bỏ 🔴)

### Immune System: 4 phiên bản (v1→v4) — 57 tests

| v | Lớp bảo vệ | Mô tả |
|---|-----------|-------|
| v1 | Write zones | LOCKED/GUARDED/FLUID + fail-safe + strip [SYSTEM] blocks |
| v2 | CJK net + 🔴 exact + source filter | Dấu câu CJK, decision 🔴 lệch 1 ký tự = reject, decision chỉ từ user |
| v3 | Full lang net + negation guard + FLUID semantic | Quét parsed fields, giữ "KHÔNG" khi nén, current_state không mâu thuẫn LOCKED |
| **v4** | **Bytes-frozen global_context** | KDM-seeded capsule: global_context bytes-equal — thêm/bớt/sửa 1 ký tự = reject |

### Định luật Sprint B (experience_matrix)
1. Capsule inject không luật thi hành = tài liệu tham khảo; model càng mạnh phá càng thuyết phục
2. Hiến pháp phải trói CẢ kẻ nói (Actor) lẫn kẻ ghi (Compactor)
3. Ký ức phải phân vùng ghi: FROZEN (bytes-equal) / GUARDED (append + source filter) / FLUID (có lính gác ngữ nghĩa)
4. "Phép nén đánh rơi chữ KHÔNG" — nén lời từ chối phải giữ phủ định
5. Chất độc luôn tìm vùng không gác: đảo (nói dối) → cộng (pha loãng) → vùng FLUID
6. Validate-or-keep-old: ký ức cũ đúng hơn ký ức hỏng
7. Rút quyền viết thắng thêm lính gác — vùng không cửa không cần khóa
8. Bài test bẫy tự nó đầu độc ký ức — mọi lần thử lửa phải có tẩy độc + khám nghiệm hậu kỳ

### Sổ nợ v1.2 (không blocker)
- Compactor tự phong 🔴 — decision do Compactor sinh phải cap 🟡; 🔴 chỉ từ KDM hoặc user confirm
- Rác turn thô trong current_state — strip pattern "(turn: ...)"
- Ô "Lý do" bắt buộc decision + model đọc lại xác nhận
- Topic slug ≤50 ký tự (verify bên KDM)
- Learn Mode UI, kéo thả map tay, auth
