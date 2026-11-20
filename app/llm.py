"""Optional chat helper retained from the official starter."""
import os
from pathlib import Path
import httpx

def _load_env():
    path = Path(".env")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

def settings():
    _load_env()
    return (os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1", os.environ.get("LLM_API_KEY", ""), os.environ.get("LLM_MODEL", ""))

def chat(messages, **options):
    base, key, model = settings()
    if not model: raise RuntimeError("Set LLM_MODEL in .env (see .env.example).")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = httpx.post(f"{base.rstrip('/')}/chat/completions", headers=headers, timeout=120, json={"model": model, "messages": messages, **options})
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
