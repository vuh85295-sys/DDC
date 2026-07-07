"""
DCC Demo — 3-turn conversation where memory persists and evolves
without storing any raw chat history.

Requirements:
    ollama pull qwen2.5:7b-instruct     (or edit MAIN_MODEL)
    ollama pull qwen2.5:1.5b-instruct   (compactor)
    ollama pull nomic-embed-text        (embeddings)

Run:
    python demo_cli.py            # scripted 3-turn showcase
    python demo_cli.py --chat     # interactive REPL
"""

import argparse
import json
import sys

from dcc_middleware import ContextCompactorMiddleware

MAIN_MODEL = "qwen2.5:7b-instruct"
COMPACTOR_MODEL = "qwen2.5:1.5b-instruct"
EMBED_MODEL = "nomic-embed-text"
TOPIC_ID = "demo_todo_api"


def build_dcc() -> ContextCompactorMiddleware:
    return ContextCompactorMiddleware(
        main_model=MAIN_MODEL,
        compactor_model=COMPACTOR_MODEL,
        embed_model=EMBED_MODEL,
        persist_dir="./dcc_memory",
        on_event=lambda msg: print(f"  \033[90m{msg}\033[0m"),
    )


def show_capsule(dcc: ContextCompactorMiddleware) -> None:
    capsule = dcc.vault.get(TOPIC_ID)
    if capsule:
        print("\n\033[96m── Current Memory Capsule ──\033[0m")
        print(json.dumps(json.loads(capsule.to_json()), indent=2)[:1200])
        print()


def scripted_demo() -> None:
    dcc = build_dcc()
    turns = [
        "We're building a FastAPI todo API. Decide the stack: I want SQLite "
        "via SQLAlchemy and pydantic v2 models. Confirm the plan briefly.",
        "Write the Todo model and the POST /todos endpoint only.",
        "Now add GET /todos with a 'completed' filter. Keep it consistent "
        "with what we already decided.",
    ]
    for i, prompt in enumerate(turns, 1):
        print(f"\n\033[93m[Turn {i}] USER:\033[0m {prompt}")
        response = dcc.chat(TOPIC_ID, prompt)
        print(f"\033[92m[Turn {i}] LLM:\033[0m {response[:600]}"
              f"{'…' if len(response) > 600 else ''}")
        show_capsule(dcc)

    print("Done. Close and re-run — the capsule at ./dcc_memory/ persists;\n"
          "turn 4 will continue with full continuity and zero raw history.")


def interactive() -> None:
    dcc = build_dcc()
    print(f"DCC interactive mode — topic '{TOPIC_ID}'. "
          "Type /capsule to inspect memory, /quit to exit.")
    while True:
        try:
            prompt = input("\n\033[93myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt == "/capsule":
            show_capsule(dcc)
            continue
        print(f"\033[92mllm>\033[0m {dcc.chat(TOPIC_ID, prompt)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true",
                        help="interactive REPL instead of scripted demo")
    args = parser.parse_args()
    try:
        interactive() if args.chat else scripted_demo()
    except ConnectionError:
        sys.exit("Cannot reach Ollama at localhost:11434 — is it running?")
