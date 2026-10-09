"""Optional local chat model for the Ask assistant: Gemma 4 E2B served by Ollama on this computer.

Only loopback addresses are accepted, so questions and photos never leave the machine. When Ollama is not running
or the model is not pulled, the assistant still answers (counts and search) with fixed sentences.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import httpx

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:e2b-it-qat"


class ChatUnavailable(RuntimeError):
    pass


def require_loopback(url: str) -> str:
    u = urlparse(url)
    host = u.hostname or ""
    try:
        ok = ipaddress.ip_address(host).is_loopback
    except ValueError:
        ok = host == "localhost"
    if u.scheme != "http" or not ok:
        raise ValueError(f"the chat model must run on this computer (http://127.0.0.1:…), not {url!r}")
    return url.rstrip("/")


class OllamaChat:
    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, timeout: float = 120.0):
        self.url = require_loopback(url)
        self.model = model
        self.timeout = timeout

    def status(self) -> dict:
        try:
            tags = httpx.get(self.url + "/api/tags", timeout=1.5).json()
        except (httpx.HTTPError, ValueError):
            return {"available": False, "model": self.model,
                    "reason": "Ollama is not running on this computer (see docs/ask.md)"}
        names = {m.get("name") for m in tags.get("models", [])}
        if self.model not in names:
            return {"available": False, "model": self.model,
                    "reason": f"the chat model is not downloaded yet: run `ollama pull {self.model}`"}
        return {"available": True, "model": self.model}

    def chat(self, messages: list[dict], *, schema: dict | None = None, max_tokens: int = 200) -> str:
        body = {"model": self.model, "messages": messages, "stream": False, "think": False, "keep_alive": "5m",
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096}}
        if schema is not None:
            body["format"] = schema
        try:
            r = httpx.post(self.url + "/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise ChatUnavailable(f"the local chat model did not answer: {e}") from e

